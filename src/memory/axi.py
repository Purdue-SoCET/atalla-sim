"""The AXI bus between the DRAM masters and DRAM: the memory_FA26 branch's
split-transaction interconnect (rtl/modules/memory/axi_bus) and the
Ramulator wrapper at its end (rtl/modules/memory/ramulator_wrapper).

Six masters -- the four scratchpad backends (SP0-SP3, ids 0-3), the data
cache (4) and the instruction cache (5; reads only) -- each hold an AXIPort.
A read is one AR of up to 16 beats of 8 bytes; a write is one AW and its W
beats. The bus has a read path and a write path that run side by side:

    read   AR FIFO per master (4) -> round-robin arbiter, one AR a cycle ->
           wrapper (8192 burst slots): every beat goes to the DRAM model at
           once -> beats come back in AR order, one a cycle on the shared R
           channel, oldest burst first (single-beat reads fill the gaps)
    write  AW FIFO (3) and W FIFO (31 beats) per master -> round-robin
           arbiter, held for a whole burst -> driver, one burst at a time ->
           wrapper: one 8-byte DRAM write per beat -> B when the last beat
           is taken

The cycles, with every ready high (axi_read*.sv, axi_write_*.sv,
ramulator_sv_wrapper.sv):

    read   AR handshake in t; the wrapper takes it in t + 1; a beat the DRAM
           completes in cycle c reaches the master in c + 2
    write  AW and W0 in t (then one W beat a cycle); the wrapper takes AW and
           W0 in t + 4 and a beat a cycle after; the last in F = t + 4 + len;
           B reaches the master in F + 1. The next burst's first beat goes in
           F + 3, or F + 4 if its master is the one that just finished and no
           other is waiting (the arbiter skips the owner at its pop).

B comes back when the wrapper has taken the beats, not when the DRAM has
written them. Two things the RTL does are left out: the wrapper's registered
ARREADY and WREADY with its accept-on-valid (an AR it refuses can be popped,
and W beats are taken twice), and its 256-bit beats packed onto the 64-bit
bus. The model takes one AR or W beat when it can, as intended.

The DRAM behind the wrapper is any object with send(cycle, addr, nbytes,
is_write, on_done, coalesce) -> bool and a tick after the bus: RamulatorDRAM
(memory/ramulator.py) or FixedLatencyDRAM below. The bus ticks after the
masters (PHASE_AXI) and before the DRAM (PHASE_DRAM).
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional

from base.clocked_object import Clocked

BEAT_BYTES = 8
MAX_BEATS = 16                 # AxLEN is 4 bits
MASTERS = ("sp0", "sp1", "sp2", "sp3", "dcache", "icache")
WRITE_MASTERS = MASTERS[:5]


@dataclass
class AXIParams:
    """The bus's sizes (axi_bus_pkg.sv), all changeable."""
    ar_depth: int = 4          # AR FIFO per master (extra-MSB pointers: all usable)
    aw_depth: int = 3          # AW FIFO per master: 4 entries, one never used
    w_depth: int = 31          # W FIFO per master: NUM_U_WRITES * 8, one never used
    driver_beats: int = 4      # the write driver's beat FIFO
    drain_per_cycle: int = 8   # DRAM completions the wrapper takes a cycle
    burst_slots: int = 8192    # wrapper's N_BURST_SLOTS: read bursts taken, not yet returned
    single_depth: int = 8192   # wrapper's SR_DEPTH: single-beat data waiting for R
    #: Bytes a beat carries: the bus's 64 bits. The wrapper's own benchmark
    #: (tb/unit/memory/ramulator_wrapper/test_sdma.sv) drives it with 32.
    beat_bytes: int = BEAT_BYTES


@dataclass(eq=False)
class _Read:
    mid: int
    addr: int
    beats: int
    on_beat: Callable[[int, int], None]
    visible: int               # first cycle the arbiter can see it
    sent: int = 0              # beats handed to the DRAM
    ready: List[bool] = field(default_factory=list)
    out: int = 0               # beats sent to the master


@dataclass(eq=False)
class _Write:
    mid: int
    addr: int
    beats: int
    on_accept: Optional[Callable[[int, int], None]]
    on_done: Optional[Callable[[int], None]]
    pushed: int                # cycle W beat 0 entered the W FIFO; beat i at pushed + i
    loaded: int = 0            # beats the driver has taken from the W FIFO


class AXIPort:
    """One master's side of the bus. Requests made in a cycle reach the bus's
    FIFOs that cycle and its arbiters the next."""

    #: Backends and cache ports check this to speak AXI instead of bursts.
    axi = True

    def __init__(self, bus: "AXIBus", mid: int, name: str):
        self.bus, self.mid, self.name = bus, mid, name
        self.reads: Deque[_Read] = deque()       # AR FIFO
        self.writes: Deque[_Write] = deque()     # AW FIFO, head = burst in the driver
        self.w_beats = 0                         # beats in the W FIFO
        self.w_next = 0                          # cycle the master's W channel is free
        self.stats = dict(reads=0, writes=0, read_beats=0, write_beats=0, ar_full=0,
                          aw_full=0)

    @property
    def last_issue_tick(self) -> int:
        """The last cycle the bus handed a request to the DRAM."""
        return self.bus.last_issue_tick

    def can_read(self) -> bool:
        if len(self.reads) < self.bus.params.ar_depth:
            return True
        self.stats["ar_full"] += 1
        return False

    def read(self, cycle: int, addr: int, beats: int,
             on_beat: Callable[[int, int], None]) -> None:
        """Read `beats` 8-byte beats from `addr`; on_beat(i, c) says beat i
        is the master's from cycle c."""
        assert 0 < beats <= MAX_BEATS and self.can_read()
        self.reads.append(_Read(self.mid, int(addr), int(beats), on_beat, int(cycle) + 1,
                                ready=[False] * int(beats)))
        self.stats["reads"] += 1
        self.stats["read_beats"] += beats
        self.bus.wake()

    def can_write(self, beats: int) -> bool:
        if (len(self.writes) < self.bus.params.aw_depth
                and self.w_beats + beats <= self.bus.params.w_depth):
            return True
        self.stats["aw_full"] += 1
        return False

    def write(self, cycle: int, addr: int, beats: int,
              on_accept: Optional[Callable[[int, int], None]] = None,
              on_done: Optional[Callable[[int], None]] = None) -> None:
        """Write `beats` beats to `addr`, the master sending one W beat a
        cycle. on_accept(i, c): the wrapper took beat i in cycle c (the data
        reaches memory then); on_done(c): B is the master's from cycle c."""
        assert self.mid < len(WRITE_MASTERS) and 0 < beats <= MAX_BEATS
        assert self.can_write(beats)
        start = max(int(cycle), self.w_next)
        self.w_next = start + beats
        self.writes.append(_Write(self.mid, int(addr), int(beats), on_accept, on_done, start))
        self.w_beats += beats
        self.stats["writes"] += 1
        self.stats["write_beats"] += beats
        self.bus.wake()

    @property
    def idle(self) -> bool:
        return not self.reads and not self.writes


class AXIBus(Clocked):
    """The interconnect and the wrapper; see the module docstring."""

    def __init__(self, dram, params: Optional[AXIParams] = None):
        super().__init__()
        self.dram = dram
        self.params = params or AXIParams()
        self.ports = [AXIPort(self, mid, name) for mid, name in enumerate(MASTERS)]
        self.by_name: Dict[str, AXIPort] = {p.name: p for p in self.ports}
        self.last_issue_tick = -1
        # read side
        self._r_last: Optional[int] = None       # round-robin pointer, None = idle
        self._r_locked: Optional[int] = None     # master held through a refused AR
        self._bursts: Deque[_Read] = deque()     # taken, in AR order (len > 0)
        self._single: Deque[tuple] = deque()     # completed single-beat reads
        self._arrived: Deque[tuple] = deque()    # DRAM completions not yet drained
        # write side (arbiter, driver, wrapper registers)
        self._w_granted = False
        self._w_owner = len(WRITE_MASTERS) - 1   # first scan starts at SP0
        self._w_count = 0
        self._d_burst: Optional[_Write] = None   # burst in the driver
        self._d_aw = False                       # its AW is in the driver's buffer
        self._d_aw_sent = False
        self._d_fifo: Deque[int] = deque()       # beat indices in the driver's FIFO
        self._d_fired = 0
        self.stats = dict(ar=0, aw=0, r_beats=0, w_beats=0, dram_refused=0)

    def port(self, name: str) -> AXIPort:
        return self.by_name[name]

    # -- read path ----------------------------------------------------------------------
    def _beat_addr(self, r: _Read, i: int) -> int:
        return r.addr + i * self.params.beat_bytes

    def _send_read_beat(self, r: _Read, cycle: int) -> bool:
        i = r.sent
        ok = self.dram.send(cycle, self._beat_addr(r, i), self.params.beat_bytes, False,
                            lambda c, r=r, i=i: self._arrive(r, i),
                            coalesce=r.beats > 1)
        if ok:
            r.sent += 1
            self.last_issue_tick = cycle
        else:
            self.stats["dram_refused"] += 1
        return ok

    def _arrive(self, r: _Read, i: int) -> None:
        self._arrived.append((r, i))
        self.wake()

    def _read_path(self, cycle: int) -> None:
        # 1. drain DRAM completions
        for _ in range(min(self.params.drain_per_cycle, len(self._arrived))):
            r, i = self._arrived.popleft()
            if r.beats > 1:
                r.ready[i] = True
            else:
                self._single.append((r, i))
        # 1b. one R beat: the oldest burst's next beat, else a single-beat read
        head = self._bursts[0] if self._bursts else None
        if head is not None and head.ready[head.out]:
            head.on_beat(head.out, cycle + 1)
            head.out += 1
            if head.out == head.beats:
                self._bursts.popleft()
            self.stats["r_beats"] += 1
        elif self._single:
            r, i = self._single.popleft()
            r.on_beat(i, cycle + 1)
            self.stats["r_beats"] += 1
        # 2. beats the DRAM refused earlier
        for r in self._bursts:
            if r.sent < r.beats:
                self._send_read_beat(r, cycle)
        # 3. the arbiter's AR, if the wrapper can take its first beat
        n = len(self.ports)
        req = [bool(p.reads) and p.reads[0].visible <= cycle for p in self.ports]
        if not any(req):
            self._r_last = self._r_locked = None
            return
        if self._r_locked is not None and req[self._r_locked]:
            sel = self._r_locked
        else:
            start = 0 if self._r_last is None else self._r_last + 1
            sel = next((start + k) % n for k in range(n) if req[(start + k) % n])
        r = self.ports[sel].reads[0]
        room = (len(self._bursts) < self.params.burst_slots if r.beats > 1
                else len(self._single) < self.params.single_depth)
        if room and self._send_read_beat(r, cycle):
            self.ports[sel].reads.popleft()
            self._r_last, self._r_locked = sel, None
            self.stats["ar"] += 1
            if r.beats > 1:
                self._bursts.append(r)
                while r.sent < r.beats and self._send_read_beat(r, cycle):
                    pass
        else:
            self._r_locked = sel

    # -- write path ---------------------------------------------------------------------
    def _w_req(self, cycle: int) -> List[bool]:
        return [bool(p.writes) and p.writes[0].pushed < cycle for p in self.ports[:5]]

    def _write_path(self, cycle: int) -> None:
        n = len(WRITE_MASTERS)
        req = self._w_req(cycle)
        b = self._d_burst
        # combinational: driver loads, subordinate handshake
        load_aw = (b is None and self._w_granted and req[self._w_owner])
        load_w = (b is not None and b.loaded < b.beats
                  and b.pushed + b.loaded < cycle                 # beat in the W FIFO
                  and len(self._d_fifo) < self.params.driver_beats)
        w_fire = False
        if self._d_fifo and (self._d_aw_sent or self._d_aw):
            i = self._d_fifo[0]
            ok = self.dram.send(cycle, b.addr + i * self.params.beat_bytes,
                                self.params.beat_bytes, True, lambda c: None)
            if ok:
                w_fire = True
                self.last_issue_tick = cycle
                if b.on_accept is not None:
                    b.on_accept(i, cycle)
            else:
                self.stats["dram_refused"] += 1
        # arbiter (registered owner, burst-atomic)
        next_granted, next_owner, next_count = self._w_granted, self._w_owner, self._w_count
        skid_ready = b is None
        pop = None
        if not self._w_granted:
            if skid_ready:
                for k in range(1, n + 1):
                    m = (self._w_owner + k) % n
                    if req[m]:
                        next_granted, next_owner = True, m
                        next_count = self.ports[m].writes[0].beats - 1
                        break
        elif w_fire:
            if self._w_count:
                next_count = self._w_count - 1
            else:
                pop = self._w_owner
                next_granted = False
                for k in range(1, n):
                    m = (self._w_owner + k) % n
                    if req[m]:
                        next_granted, next_owner = True, m
                        next_count = self.ports[m].writes[0].beats - 1
                        break
        # registers
        if load_aw:
            b = self._d_burst = self.ports[self._w_owner].writes[0]
            self._d_aw, self._d_aw_sent, self._d_fired = True, False, 0
            self.stats["aw"] += 1
        if load_w:
            self._d_fifo.append(b.loaded)
            b.loaded += 1
            self.ports[b.mid].w_beats -= 1
        if w_fire:
            if self._d_aw:
                self._d_aw, self._d_aw_sent = False, True
            self._d_fifo.popleft()
            self._d_fired += 1
            self.stats["w_beats"] += 1
            if self._d_fired == b.beats:
                self._d_burst = None
                if b.on_done is not None:
                    b.on_done(cycle + 1)
        if pop is not None:
            self.ports[pop].writes.popleft()
        self._w_granted, self._w_owner, self._w_count = next_granted, next_owner, next_count

    # -- clock --------------------------------------------------------------------------
    def tick(self, time=None) -> None:
        cycle = self._consume_tick(time)
        if cycle is None:
            return
        self._read_path(cycle)
        self._write_path(cycle)

    def summary(self, cycles: int) -> str:
        s, cyc = self.stats, max(1, cycles)
        lines = ["AXI bus: %d read bursts, %d write bursts" % (s["ar"], s["aw"]),
                 "  R %d beats (%.2f B/cycle), W %d beats (%.2f B/cycle), of 8 each"
                 % (s["r_beats"], self.params.beat_bytes * s["r_beats"] / cyc, s["w_beats"],
                    self.params.beat_bytes * s["w_beats"] / cyc)]
        if s["dram_refused"]:
            lines.append("  %d requests the DRAM refused for a full queue" % s["dram_refused"])
        full = {p.name: p.stats["ar_full"] + p.stats["aw_full"] for p in self.ports}
        if any(full.values()):
            lines.append("  cycles a master waited for a full FIFO: " + ", ".join(
                "%s %d" % kv for kv in full.items() if kv[1]))
        return "\n".join(lines)

    @property
    def idle(self) -> bool:
        return (all(p.idle for p in self.ports) and not self._bursts and not self._single
                and not self._arrived and self._d_burst is None and not self._w_granted)

    def next_wake(self, now: int) -> Optional[int]:
        return None if self.idle else now + 1


class FixedLatencyDRAM(Clocked):
    """A DRAM with no queues behind the bus: a request sent in cycle t
    completes in t + latency (latency 0: the same cycle, after the bus)."""

    models_latency = True

    def __init__(self, latency: int = 6):
        super().__init__()
        self.latency = int(latency)
        self._due: Dict[int, list] = {}
        self.stats = dict(reads=0, writes=0, refused=0, coalesced=0)

    def send(self, tick, addr, nbytes, is_write, on_done, coalesce=False) -> bool:
        self._due.setdefault(int(tick) + self.latency, []).append(on_done)
        self.stats["writes" if is_write else "reads"] += 1
        self.wake()
        return True

    def tick(self, time=None) -> None:
        cycle = self._consume_tick(time)
        if cycle is None:
            return
        for c in sorted(k for k in self._due if k <= cycle):
            for on_done in self._due.pop(c):
                on_done(cycle)

    @property
    def idle(self) -> bool:
        return not self._due

    def next_wake(self, now: int) -> Optional[int]:
        return max(now + 1, min(self._due)) if self._due else None
