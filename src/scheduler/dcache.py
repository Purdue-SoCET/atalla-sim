"""Data cache for the scalar core: a lockup-free cache with MSHRs.

This is the cache the RTL's `rtl/modules/scheduler/dcache/` sets out to be:
write-back, write-allocate, set-associative, with miss status holding
registers so that hits are served while misses are outstanding. It is not a
transcription of that RTL: the RTL's known bugs are left out, and the
geometry is configurable. The RTL's structure is the default, with the SRAM
latencies PCACTI recommends for its macro as built at 800 MHz (read 1,
write 4; the RTL instance still sets read 2). With one request at a time:

    load hit          response 2 cycles after the request is taken
    store hit         response 6 cycles after (lookup 2, SRAM write 4)
    load miss         filled 24 cycles after the miss (32 with a dirty victim)

Structure
---------
One SRAM holds the tags, state and data of every set (`sram_bank`, read 1 /
write 4 by default). Two state machines share it:

  * the **lookup** FSM takes a request, reads its set, and in the following
    EVAL cycle answers it:
      - **hit**: load data, or a store's write to the SRAM;
      - **miss**: allocate an MSHR entry, or add a target to the entry already
        open for the block. A store miss is answered at once (STORE_ACK),
        because the store's data is held in the MSHR; a load miss is answered
        MISS, and FILL with its data when the fill completes;
      - **retry**: no entry or target slot is free. The requester presents
        the request again later.
  * the **fill** FSM takes MSHR entries oldest first:
      1. read the set and pick a way (the block's own way if it is already
         present, else an invalid way, else tree pseudo-LRU);
      2. write the set back with that way invalid and capture the victim;
      3. burst the line in from memory;
      4. write the victim back if it is dirty;
      5. write the line with the entry's targets applied in order;
      6. answer every load target in one cycle.
    A block that is already present needs no burst and no victim.

The lookup FSM has priority on the SRAM. An enable is raised only when the
SRAM takes it (`!busy`), and a done goes to whichever FSM issued the read.
A fill can't be starved: a lookup leaves the SRAM free in the cycle its read
completes.

Ordering
--------
Lookups are serial, so their EVAL order is the order requests take effect. A
lookup to a block with an open MSHR entry joins that entry even if the array
holds the block, so it can't overtake older targets. An entry is open until
its line write is issued; a miss after that opens a new entry, which finds
the block present. A lookup that hits the line being evicted is treated as a
miss, so it can't write a line whose data is already captured.

All of the fill FSM's writes are way-masked: they change only their own way,
so a store hit to another way of the set is never overwritten.

Memory
------
The memory behind the cache is outside the scheduler. With a `port`
(memory/bus.py) the line fill and the writebacks go out as bursts on the
DRAM channel the scratchpad backends share, and a beat is taken when its
burst has returned; written beats wait while the port's buffer is full.
Without one, the timing is a parameter, as for the icache: a read burst
delivers its first beat `first_beat_wait` cycles after the request (0: in
the request cycle, as `sim_ram_rr` does when free), then one beat per cycle,
and each written beat takes `1 + write_wait` cycles.

Halt
----
With `halt` high and no misses outstanding, the fill FSM walks every set
and writes back the dirty lines, then raises `flushed` until `halt` falls.
"""

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Tuple

from base.rtl_module import RTLModule
from memory.sram_bank import SramChannel

WORD_BYTES = 4


# ---------------------------------------------------------------------------
# Configuration and records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DCacheConfig:
    """The data cache's size and timing. Defaults are the RTL's.
    Checked on construction: ways, sets and beats must divide evenly.
    """
    size_bytes: int = 4096        # rtl: CACHE_SIZE 1024 words
    ways: int = 4                 # NUM_WAYS
    line_bytes: int = 64          # BLOCK_SIZE 16 words
    mshrs: int = 8                # MSHR_BUFFER_LEN
    targets_per_mshr: int = 8     # requests one entry can hold
    sram_read_latency: int = 1    # PCACTI, lockup_free_cache.sv's macro as built
    sram_write_latency: int = 4
    beat_bytes: int = 8           # 64-bit memory interface
    first_beat_wait: int = 0      # memory timing (outside the scheduler)
    write_wait: int = 0

    def __post_init__(self):
        if self.ways < 1 or self.ways & (self.ways - 1):
            raise ValueError("ways must be a power of two")
        if self.line_bytes % self.beat_bytes or self.beat_bytes % WORD_BYTES:
            raise ValueError("line must be whole beats, beats whole words")
        if self.size_bytes % (self.ways * self.line_bytes):
            raise ValueError("size must be whole sets")
        if self.sets & (self.sets - 1):
            raise ValueError("set count must be a power of two")
        if self.mshrs < 1 or self.targets_per_mshr < 1:
            raise ValueError("need at least one MSHR and one target")

    @property
    def sets(self) -> int:
        return self.size_bytes // (self.ways * self.line_bytes)

    @property
    def line_words(self) -> int:
        return self.line_bytes // WORD_BYTES

    @property
    def beats(self) -> int:
        return self.line_bytes // self.beat_bytes

    @property
    def beat_words(self) -> int:
        return self.beat_bytes // WORD_BYTES

    def block(self, addr: int) -> int:
        return int(addr) // self.line_bytes

    def set_of(self, addr: int) -> int:
        return self.block(addr) % self.sets

    def tag_of(self, addr: int) -> int:
        return self.block(addr) // self.sets

    def word_of(self, addr: int) -> int:
        return (int(addr) % self.line_bytes) // WORD_BYTES


@dataclass(frozen=True)
class DCacheReq:
    """A word load or store. `id` comes back on the response."""
    id: object
    addr: int
    store: bool = False
    data: int = 0


HIT, MISS, FILL, STORE_ACK, RETRY = "hit", "miss", "fill", "store_ack", "retry"


@dataclass(frozen=True)
class DCacheResp:
    """The cache's answer to a DCacheReq: a hit, a miss (the data
    follows later as a FILL), a store's acknowledgement, or RETRY (no MSHR free:
    send it again).
    """
    id: object
    addr: int
    store: bool
    kind: str                     # HIT, MISS, FILL, STORE_ACK or RETRY
    data: int = 0                 # loads: the word


@dataclass(frozen=True)
class Frame:
    """One way of one set."""
    valid: bool = False
    dirty: bool = False
    tag: int = 0
    words: Tuple[int, ...] = ()


@dataclass
class MshrEntry:
    """A miss being served: the line it fetches and the requests waiting for it."""
    uid: int
    block: int
    targets: List[DCacheReq] = field(default_factory=list)
    open: bool = True             # takes targets until its line is written


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class WordMemory:
    """Backing store, a word per 4-byte address. Unwritten words read as
    `init(addr)` (0 by default)."""

    def __init__(self, init: Optional[Callable[[int], int]] = None):
        self.words: Dict[int, int] = {}
        self.init = init or (lambda addr: 0)

    def read(self, addr: int) -> int:
        a = int(addr) & ~3
        return self.words.get(a, self.init(a)) & 0xFFFFFFFF

    def write(self, addr: int, value: int) -> None:
        self.words[int(addr) & ~3] = int(value) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Tree pseudo-LRU (cache_bank.sv's tree_based_lru_update / LRU_CALC)
# ---------------------------------------------------------------------------

def plru_victim(tree: List[int], ways: int) -> int:
    node, way = 0, 0
    for _ in range(ways.bit_length() - 1):
        bit = tree[node]
        way = (way << 1) | bit
        node = 2 * node + 1 + bit
    return way


def plru_touch(tree: List[int], ways: int, way: int) -> List[int]:
    tree = list(tree)
    levels = ways.bit_length() - 1
    node = 0
    for level in range(levels):
        bit = (way >> (levels - 1 - level)) & 1
        tree[node] = bit ^ 1              # point away from the way used
        node = 2 * node + 1 + bit
    return tree


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------

# lookup FSM
L_IDLE, L_WAIT_READ, L_EVAL, L_WAIT_WRITE = "idle", "wait_read", "eval", "wait_write"
# fill FSM
F_IDLE, F_READ, F_READ_WAIT, F_CALC, F_LRU, F_BUFFER = (
    "idle", "read", "read_wait", "calc", "lru", "buffer")
F_INVAL, F_INVAL_WAIT, F_PULL, F_EJECT = "inval", "inval_wait", "pull", "eject"
F_WRITE, F_WRITE_WAIT, F_DONE = "write", "write_wait", "done"
F_FL_READ, F_FL_WAIT, F_FL_WAY, F_FL_WB, F_FL_WRITE, F_FL_WRITE_WAIT, F_HALTED = (
    "flush_read", "flush_wait", "flush_way", "flush_wb", "flush_write",
    "flush_write_wait", "halted")

LOOKUP, FILLER = "lookup", "fill"


class DCache(RTLModule):
    """See the module docstring.

    Ports, per cycle:
      in_req_valid, in_req     a DCacheReq, held by the requester until taken
      in_halt                  flush and stop
      out_req_ready            in_req is taken this cycle if valid
      out_resp                 list of DCacheResp produced this cycle
      out_flushed              halted with every dirty line written back
      out_mem                  the memory port this cycle, for tracing:
                               ('read', block address, beat) / ('write', addr)

    State changes are collected in eval_data and applied in commit, so
    nothing evaluated this cycle sees another part's next state.
    """

    INS = dict(req_valid=False, req=None, halt=False)
    OUTS = dict(req_ready=False, resp=[], flushed=False, mem=None)
    CLEAR_ON_COMMIT = ("in_req_valid", "in_req")

    def __init__(self, name: str = "dcache", config: DCacheConfig = DCacheConfig(),
                 memory: Optional[WordMemory] = None, port=None):
        super().__init__(name)
        self.cfg = config
        #: memory/bus.py BusMaster, or None for the parameterised timing.
        self.port = port
        self.memory = memory if memory is not None else WordMemory()
        c = config
        self.array: List[List[Frame]] = [[Frame() for _ in range(c.ways)]
                                         for _ in range(c.sets)]
        self.lru: List[List[int]] = [[0] * (c.ways - 1) for _ in range(c.sets)]
        self.read = SramChannel(c.sram_read_latency)
        self.write = SramChannel(c.sram_write_latency)
        self.rdata: Optional[List[Frame]] = None     # latched on a read enable
        self.read_owner: Optional[str] = None
        self.mshrs: List[MshrEntry] = []
        self._uid = 0

        # lookup FSM
        self.l_state = L_IDLE
        self.l_req: Optional[DCacheReq] = None
        self.l_row: Optional[List[Frame]] = None
        self.l_way = 0
        # fill FSM
        self.f_state = F_IDLE
        self.f_entry: Optional[MshrEntry] = None
        self.f_set = 0
        self.f_row: Optional[List[Frame]] = None
        self.f_way = 0
        self.f_present = False
        self.f_victim: Optional[Frame] = None        # captured when evicted
        self.f_evicting: Optional[Tuple[int, int, int]] = None   # set, way, tag
        self.f_words: List[int] = []
        self.f_beat = 0
        self.f_wait = 0
        self.f_loads: List[DCacheResp] = []
        self.f_flush_way = 0

        self._actions: List[Callable[[], None]] = []
        self.cycle = 0
        self.stats = dict(requests=0, hits=0, misses=0, secondary=0,
                          retries=0, fills=0, present_fills=0, evictions=0,
                          writebacks=0, flush_writebacks=0, max_mshrs=0,
                          fill_wait_cycles=0, mem_wait_cycles=0)

    # -- helpers ----------------------------------------------------------
    @property
    def sram_busy(self) -> bool:
        return self.read.busy or self.write.busy

    def idle(self) -> bool:
        return (self.l_state == L_IDLE and self.f_state in (F_IDLE, F_HALTED)
                and not self.mshrs and not self.sram_busy)

    def _later(self, fn: Callable[[], None]) -> None:
        self._actions.append(fn)

    def _open_entry(self, block: int) -> Optional[MshrEntry]:
        for e in self.mshrs:
            if e.block == block and e.open:
                return e
        return None

    def _issue_read(self, owner: str, set_idx: int) -> None:
        self._sram_claimed = True

        def act():
            self.rdata = list(self.array[set_idx])
            self.read_owner = owner
            assert self.read.edge(True)
        self._read_en = True
        self._later(act)

    def _issue_write(self, set_idx: int, way: int, frame_fn: Callable[[], Frame],
                     on_edge: Optional[Callable[[], None]] = None) -> None:
        """A way-masked write. `frame_fn` builds the frame at the edge, from
        the array as the edge finds it."""
        self._sram_claimed = True

        def act():
            if on_edge:
                on_edge()
            self.array[set_idx][way] = frame_fn()
            assert self.write.edge(True)
        self._write_en = True
        self._later(act)

    def _read_done(self, owner: str) -> bool:
        return self.read.done and self.read_owner == owner

    def _touch(self, set_idx: int, way: int) -> None:
        self._later(lambda: self.lru.__setitem__(
            set_idx, plru_touch(self.lru[set_idx], self.cfg.ways, way)))

    # -- RTLModule phases ---------------------------------------------------
    def eval_ready(self) -> None:
        flushing = self.f_state in (F_FL_READ, F_FL_WAIT, F_FL_WAY, F_FL_WB,
                                    F_FL_WRITE, F_FL_WRITE_WAIT, F_HALTED)
        self.out_req_ready = (self.l_state == L_IDLE and not self.sram_busy
                              and not flushing)

    def eval_data(self) -> None:
        self._actions = []
        self._sram_claimed = False
        self._read_en = self._write_en = False
        self.out_resp = []
        self.out_mem = None
        self.out_flushed = False
        new_entry = self._eval_lookup()
        self._eval_fill(new_entry)

    def commit(self) -> None:
        for act in self._actions:
            act()
        if not self._read_en:
            self.read.edge(False)
        if not self._write_en:
            self.write.edge(False)
        self._actions = []
        self.cycle += 1
        super().commit()

    # -- lookup FSM ------------------------------------------------------------
    def _eval_lookup(self) -> Optional[MshrEntry]:
        c = self.cfg
        st = self.l_state
        if st == L_IDLE:
            if self.in_req_valid and self.out_req_ready:
                req = self.in_req
                self.stats["requests"] += 1
                self._issue_read(LOOKUP, c.set_of(req.addr))
                self._later(lambda: self._set(l_state=L_WAIT_READ, l_req=req))
            return None

        req = self.l_req
        set_idx, tag, word = c.set_of(req.addr), c.tag_of(req.addr), c.word_of(req.addr)

        if st == L_WAIT_READ:
            if self._read_done(LOOKUP):
                row = self.rdata
                self._later(lambda: self._set(l_state=L_EVAL, l_row=row))
            return None

        if st == L_WAIT_WRITE:
            if self.write.done:
                self.out_resp.append(DCacheResp(req.id, req.addr, True, HIT))
                self._touch(set_idx, self.l_way)
                self._later(lambda: self._set(l_state=L_IDLE, l_req=None, l_row=None))
            return None

        # L_EVAL
        block = c.block(req.addr)
        entry = self._open_entry(block)
        hit_way = None
        if entry is None:
            for w, f in enumerate(self.l_row):
                if f.valid and f.tag == tag:
                    hit_way = w
            if hit_way is not None and self.f_evicting == (set_idx, hit_way, tag):
                hit_way = None                      # its data is already captured

        if hit_way is not None and not req.store:
            self.stats["hits"] += 1
            data = self.l_row[hit_way].words[word]
            self.out_resp.append(DCacheResp(req.id, req.addr, False, HIT, data))
            self._touch(set_idx, hit_way)
            self._later(lambda: self._set(l_state=L_IDLE, l_req=None, l_row=None))
            return None

        if hit_way is not None:                     # store hit
            if self.sram_busy:
                return None
            self.stats["hits"] += 1
            way = hit_way

            def frame():
                f = self.array[set_idx][way]
                words = list(f.words)
                words[word] = req.data & 0xFFFFFFFF
                return replace(f, dirty=True, words=tuple(words))
            self._issue_write(set_idx, way, frame)
            self._later(lambda: self._set(l_state=L_WAIT_WRITE, l_way=way))
            return None

        # miss
        new_entry = None
        if entry is not None:
            if len(entry.targets) >= c.targets_per_mshr:
                return self._retry(req)
            self.stats["secondary"] += 1
            self._later(lambda: entry.targets.append(req))
        else:
            if len(self.mshrs) >= c.mshrs:
                return self._retry(req)
            self.stats["misses"] += 1
            self._uid += 1
            new_entry = MshrEntry(self._uid, block, [req])
            self._later(lambda: self.mshrs.append(new_entry))
            self.stats["max_mshrs"] = max(self.stats["max_mshrs"], len(self.mshrs) + 1)
        if req.store:
            self.out_resp.append(DCacheResp(req.id, req.addr, True, STORE_ACK))
        else:
            self.out_resp.append(DCacheResp(req.id, req.addr, False, MISS))
        self._later(lambda: self._set(l_state=L_IDLE, l_req=None, l_row=None))
        return new_entry

    def _retry(self, req: DCacheReq) -> None:
        self.stats["retries"] += 1
        self.out_resp.append(DCacheResp(req.id, req.addr, req.store, RETRY))
        self._later(lambda: self._set(l_state=L_IDLE, l_req=None, l_row=None))
        return None

    def _set(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self, k, v)

    # -- fill FSM ------------------------------------------------------------
    def _next_entry(self, new_entry: Optional[MshrEntry]) -> Optional[MshrEntry]:
        """Oldest entry not yet started. A miss allocated this cycle is
        visible, as the RTL's MSHR bypass makes it."""
        pending = [e for e in self.mshrs if e is not self.f_entry and e.open]
        if new_entry is not None:
            pending.append(new_entry)
        return pending[0] if pending else None

    def _can_use_sram(self) -> bool:
        return not self._sram_claimed and not self.sram_busy

    def _eval_fill(self, new_entry: Optional[MshrEntry]) -> None:
        """One cycle of the fill FSM: run the step for its current state.
        Each _fill_<state> method below is one state of the RTL's fill FSM;
        to add a state, write its method and add it to FILL_STEPS."""
        self.FILL_STEPS[self.f_state](self, self.cfg, new_entry)

    def _fill_idle_or_done(self, c, new_entry) -> None:
        st = self.f_state
        if st == F_DONE:
            self.out_resp.extend(self.f_loads)
            self._touch(self.f_set, self.f_way)
        nxt = self._next_entry(new_entry)
        if nxt is not None:
            self._later(lambda: self._set(f_state=F_READ, f_entry=nxt,
                                          f_set=nxt.block % c.sets, f_loads=[]))
        elif (self.in_halt and not self.mshrs and new_entry is None
              and self.l_state == L_IDLE
              and not (self.in_req_valid and self.out_req_ready)):
            self._later(lambda: self._set(f_state=F_FL_READ, f_set=0,
                                          f_entry=None, f_loads=[]))
        else:
            self._later(lambda: self._set(f_state=F_IDLE, f_entry=None, f_loads=[]))

    def _fill_read(self, c, new_entry) -> None:
        if self._can_use_sram():
            self._issue_read(FILLER, self.f_set)
            self._later(lambda: self._set(f_state=F_READ_WAIT))
        else:
            self.stats["fill_wait_cycles"] += 1

    def _fill_read_wait(self, c, new_entry) -> None:
        if self._read_done(FILLER):
            row = self.rdata
            self._later(lambda: self._set(f_state=F_CALC, f_row=row))

    def _fill_calc(self, c, new_entry) -> None:
        tag = self.f_entry.block // c.sets
        present = [w for w, f in enumerate(self.f_row) if f.valid and f.tag == tag]
        way = present[0] if present else None
        self._later(lambda: self._set(f_state=F_LRU, f_present=way is not None,
                                      f_way=way if way is not None else 0))

    def _fill_lru(self, c, new_entry) -> None:
        if not self.f_present:
            invalid = [w for w, f in enumerate(self.f_row) if not f.valid]
            way = invalid[0] if invalid else plru_victim(self.lru[self.f_set], c.ways)
            self._later(lambda: self._set(f_way=way))
        self._later(lambda: self._set(f_state=F_BUFFER))

    def _fill_buffer(self, c, new_entry) -> None:
        if self.f_present:
            words = list(self.f_row[self.f_way].words)
            self._later(lambda: self._set(f_state=F_WRITE, f_words=words))
        else:
            self._later(lambda: self._set(f_state=F_INVAL))

    def _fill_inval(self, c, new_entry) -> None:
        if not self._can_use_sram():
            self.stats["fill_wait_cycles"] += 1
            return
        set_idx, way = self.f_set, self.f_way

        def capture():
            victim = self.array[set_idx][way]
            self.f_victim = victim
            self.f_evicting = (set_idx, way, victim.tag) if victim.valid else None
        self._issue_write(set_idx, way,
                          lambda: replace(self.array[set_idx][way], valid=False),
                          on_edge=capture)
        self._later(lambda: self._set(f_state=F_INVAL_WAIT))

    def _fill_inval_wait(self, c, new_entry) -> None:
        if self.write.done:
            self._later(lambda: self._set(f_state=F_PULL, f_beat=0,
                                          f_wait=c.first_beat_wait,
                                          f_words=[0] * c.line_words))
            if self.port is not None:
                base = self.f_entry.block * c.line_bytes
                self._later(lambda: self.port.read(base, c.line_bytes))

    def _fill_pull(self, c, new_entry) -> None:
        base = self.f_entry.block * c.line_bytes
        beat = self.f_beat
        if self.port is not None:
            if not self.port.beat_ready(beat):
                self.stats["mem_wait_cycles"] += 1
                self.out_mem = ("read", base, None)
                return
            self.out_mem = ("read", base, beat)
            if beat + 1 == c.beats:
                self._later(self.port.finish_read)
        else:
            self.out_mem = ("read", base, beat if self.f_wait == 0 else None)
            if self.f_wait > 0:
                self._later(lambda: self._set(f_wait=self.f_wait - 1))
                return
        words = list(self.f_words)
        for k in range(c.beat_words):
            i = beat * c.beat_words + k
            words[i] = self.memory.read(base + i * WORD_BYTES)
        if beat + 1 < c.beats:
            self._later(lambda: self._set(f_words=words, f_beat=beat + 1))
        else:
            dirty = self.f_victim is not None and self.f_victim.valid and self.f_victim.dirty
            self._later(lambda: self._set(
                f_words=words, f_beat=0, f_wait=c.write_wait,
                f_state=F_EJECT if dirty else F_WRITE))

    def _fill_eject(self, c, new_entry) -> None:
        self._eval_writeback(self.f_victim, self.f_set, F_WRITE, "writebacks")

    def _fill_write(self, c, new_entry) -> None:
        if not self._can_use_sram():
            self.stats["fill_wait_cycles"] += 1
            return
        entry, set_idx, way = self.f_entry, self.f_set, self.f_way
        tag = entry.block // c.sets
        present = self.f_present

        def build():
            # Targets in order; loads see every store before them.
            words = list(self.f_words)
            dirty = present and self.array[set_idx][way].dirty
            loads = []
            for t in entry.targets:
                w = c.word_of(t.addr)
                if t.store:
                    words[w] = t.data & 0xFFFFFFFF
                    dirty = True
                else:
                    loads.append(DCacheResp(t.id, t.addr, False, FILL, words[w]))
            self.f_loads = loads
            entry.open = False
            return Frame(True, dirty, tag, tuple(words))
        self._issue_write(set_idx, way, build)
        self.stats["fills"] += 1
        if present:
            self.stats["present_fills"] += 1
        elif self.f_victim is not None and self.f_victim.valid:
            self.stats["evictions"] += 1
        self._later(lambda: self._set(f_state=F_WRITE_WAIT))

    def _fill_write_wait(self, c, new_entry) -> None:
        if self.write.done:
            entry = self.f_entry

            def finish():
                self.mshrs.remove(entry)
                self.f_state = F_DONE
                self.f_evicting = None
                self.f_victim = None
            self._later(finish)

    # -- halt: write back every dirty line ------------------------------

    def _fill_fl_read(self, c, new_entry) -> None:
        if self._can_use_sram():
            self._issue_read(FILLER, self.f_set)
            self._later(lambda: self._set(f_state=F_FL_WAIT))

    def _fill_fl_wait(self, c, new_entry) -> None:
        if self._read_done(FILLER):
            row = self.rdata
            self._later(lambda: self._set(f_state=F_FL_WAY, f_row=row, f_flush_way=0))

    def _fill_fl_way(self, c, new_entry) -> None:
        way = self.f_flush_way
        frame = self.f_row[way]
        if frame.valid and frame.dirty:
            self._later(lambda: self._set(f_state=F_FL_WB, f_way=way,
                                          f_victim=frame, f_beat=0,
                                          f_wait=c.write_wait))
        elif way + 1 < c.ways:
            self._later(lambda: self._set(f_flush_way=way + 1))
        elif self.f_set + 1 < c.sets:
            self._later(lambda: self._set(f_state=F_FL_READ, f_set=self.f_set + 1))
        else:
            self._later(lambda: self._set(f_state=F_HALTED))

    def _fill_fl_wb(self, c, new_entry) -> None:
        self._eval_writeback(self.f_victim, self.f_set, F_FL_WRITE, "flush_writebacks")

    def _fill_fl_write(self, c, new_entry) -> None:
        if self._can_use_sram():
            set_idx, way = self.f_set, self.f_way
            self._issue_write(set_idx, way,
                              lambda: replace(self.array[set_idx][way], dirty=False))
            row = list(self.f_row)
            row[way] = replace(row[way], dirty=False)
            self._later(lambda: self._set(f_state=F_FL_WRITE_WAIT, f_row=row))

    def _fill_fl_write_wait(self, c, new_entry) -> None:
        if self.write.done:
            self._later(lambda: self._set(f_state=F_FL_WAY))

    def _fill_halted(self, c, new_entry) -> None:
        self.out_flushed = bool(self.in_halt)
        if not self.in_halt:
            self._later(lambda: self._set(f_state=F_IDLE))

    #: f_state -> the method that runs one cycle of that state.
    FILL_STEPS = {
        F_IDLE: _fill_idle_or_done,
        F_DONE: _fill_idle_or_done,
        F_READ: _fill_read,
        F_READ_WAIT: _fill_read_wait,
        F_CALC: _fill_calc,
        F_LRU: _fill_lru,
        F_BUFFER: _fill_buffer,
        F_INVAL: _fill_inval,
        F_INVAL_WAIT: _fill_inval_wait,
        F_PULL: _fill_pull,
        F_EJECT: _fill_eject,
        F_WRITE: _fill_write,
        F_WRITE_WAIT: _fill_write_wait,
        F_FL_READ: _fill_fl_read,
        F_FL_WAIT: _fill_fl_wait,
        F_FL_WAY: _fill_fl_way,
        F_FL_WB: _fill_fl_wb,
        F_FL_WRITE: _fill_fl_write,
        F_FL_WRITE_WAIT: _fill_fl_write_wait,
        F_HALTED: _fill_halted,
    }

    def _eval_writeback(self, frame: Frame, set_idx: int, then: str, stat: str) -> None:
        """One cycle of writing `frame` back, a beat at a time."""
        c = self.cfg
        base = (frame.tag * c.sets + set_idx) * c.line_bytes
        beat = self.f_beat
        self.out_mem = ("write", base + beat * c.beat_bytes)
        if self.port is not None:
            if not self.port.can_write():
                self.stats["mem_wait_cycles"] += 1
                return
            addr0 = base + beat * c.beat_bytes
            self._later(lambda: self.port.write(addr0, c.beat_bytes))
        elif self.f_wait > 0:
            self._later(lambda: self._set(f_wait=self.f_wait - 1))
            return
        for k in range(c.beat_words):
            i = beat * c.beat_words + k
            addr, value = base + i * WORD_BYTES, frame.words[i]
            self._later(lambda a=addr, v=value: self.memory.write(a, v))
        if beat + 1 < c.beats:
            self._later(lambda: self._set(f_beat=beat + 1, f_wait=c.write_wait))
        else:
            self.stats[stat] += 1
            self._later(lambda: self._set(f_beat=0, f_state=then))

    # -- convenience -----------------------------------------------------------
    def step(self, req: Optional[DCacheReq] = None, halt: bool = False):
        """One cycle: present `req` (or nothing), return (taken, responses)."""
        self.in_req_valid, self.in_req, self.in_halt = req is not None, req, halt
        self.eval_ready()
        taken = bool(self.in_req_valid and self.out_req_ready)
        self.eval_data()
        resp = list(self.out_resp)
        self.commit()
        return taken, resp

    def warm(self, addresses, memory_values: bool = True) -> None:
        """Install the lines holding these addresses, clean, as if already
        filled. A test convenience; there is no such port."""
        c = self.cfg
        for addr in addresses:
            s, t = c.set_of(addr), c.tag_of(addr)
            row = self.array[s]
            if any(f.valid and f.tag == t for f in row):
                continue
            free = [w for w, f in enumerate(row) if not f.valid]
            way = free[0] if free else plru_victim(self.lru[s], c.ways)
            base = c.block(addr) * c.line_bytes
            words = tuple(self.memory.read(base + i * WORD_BYTES) if memory_values else 0
                          for i in range(c.line_words))
            row[way] = Frame(True, False, t, words)
            self.lru[s] = plru_touch(self.lru[s], c.ways, way)
