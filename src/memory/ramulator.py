"""DRAM timing from Ramulator 2, in place of the fixed-latency channel.

The default DRAM timing (SharedDRAMBurstChannel, memory/backend.py) answers
every burst a fixed dram_latency cycles after it launches. RamulatorDRAM
hands each burst to Ramulator 2's DDR4 controller instead -- bank and row
state, refresh, FR-FCFS scheduling and a 32-entry read and write queue
each -- and a burst completes when Ramulator says so: a read when its data
returns, a write when its WR command issues.

Ramulator only times requests. The data still lives in atalla-sim's DRAM
(memory/dram.py): a backend reads or writes it when the burst completes,
as with the fixed channel.

    masters (BACKEND phase)   can_issue(), send() -> True if Ramulator took it
    this (DRAM phase, after)  tick(): runs Ramulator for this cycle and calls
                              each completed burst's on_done(cycle)

The masters see a burst completed in cycle c from cycle c + 1. One burst
launches per cycle in total, as on the fixed channel.

The configuration is named (DRAM_CONFIGS) and every value in it can be
overridden. The default, "rtl", is the memory_FA26 branch's: its AXI bus
ends in a Ramulator wrapper running DDR4-3200W (8Gb x16, 2 ranks, one
64-bit channel) at two memory cycles per 800 MHz core cycle, with 256 read
and 64 write queue entries, and reads to a 64-byte block in flight merged
into one. "ddr4-1600" and "ddr_cntrl" are the non-blocking DDR controller's
organization at DDR4-1600J, by the spec and with dram_pkg.vh's own timings.
The address mapping is Ramulator's RoBaRaCoCh.

Every DRAM request is one DDR4 transaction of 64 bytes (BL8 on 64 bits),
however few bytes it carries: an 8-byte write costs as much data-bus time
as a 64-byte one. Merged reads share theirs.

Needs `make ramulator` (a C++20 compiler and CMake); ramulator_available()
says whether it has been built.
"""

import ctypes
import os
import sys
from fractions import Fraction
from typing import Callable, Dict, Optional

from base.clocked_object import Clocked

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAMULATOR_DIR = os.path.join(_ROOT, "third_party", "ramulator2")
_LIB_PATH = os.path.join(_ROOT, "src", "native", "libatalla_ramulator.so")

#: The core clock the memory clock is compared with (PCACTI's sweeps assume it).
CORE_MHZ = 800

_lib = None
_load_error: Optional[str] = None


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    try:
        lib = ctypes.CDLL(_LIB_PATH)
    except OSError as exc:
        _load_error = "%s (build it with `make ramulator`)" % exc
        return None
    vp = ctypes.c_void_p
    lib.atr_last_error.restype = ctypes.c_char_p
    lib.atr_last_error.argtypes = []
    lib.atr_create.restype = vp
    lib.atr_create.argtypes = [ctypes.c_char_p]
    lib.atr_destroy.restype = None
    lib.atr_destroy.argtypes = [vp]
    lib.atr_tx_bytes.restype = ctypes.c_int
    lib.atr_tx_bytes.argtypes = [vp]
    lib.atr_tck_ns.restype = ctypes.c_double
    lib.atr_tck_ns.argtypes = [vp]
    lib.atr_send.restype = ctypes.c_int
    lib.atr_send.argtypes = [vp, ctypes.c_int, ctypes.c_uint64, ctypes.c_int, ctypes.c_int64]
    lib.atr_tick.restype = None
    lib.atr_tick.argtypes = [vp, ctypes.c_int64]
    lib.atr_drain.restype = ctypes.c_int
    lib.atr_drain.argtypes = [vp, ctypes.POINTER(ctypes.c_int64), ctypes.c_int]
    lib.atr_stats.restype = ctypes.c_char_p
    lib.atr_stats.argtypes = [vp]
    _lib = lib
    return lib


def ramulator_available() -> bool:
    return _load() is not None


#: Named DRAM configurations. Every field can be overridden by name
#: (ramulator_config(config, **overrides), run_kernel --dram-param), as can any
#: Ramulator DDR4 timing or organization value (nCL=12, tCK_ps=625, ...).
#:
#: "rtl" is the memory_FA26 branch's memory system: the AXI bus's subordinate
#: is the Ramulator wrapper (rtl/modules/memory/ramulator_wrapper), with
#: configs/ddr4_config.yaml -- DDR4 8Gb x16, DDR4-3200W, one channel, 2 ranks,
#: RoBaRaCoCh, FR-FCFS, all-bank refresh, open rows, 256 read and 64 write
#: queue entries -- ticked twice per 800 MHz cycle (ramulator_sv_wrapper.sv
#: step 7), so DDR4-3200's 1600 MHz memory clock runs at its own speed.
#: Reads to a 64-byte block already being read share its one DRAM read
#: (ramulator_dpi.cpp's coalescing); writes go one by one.
#:
#: "ddr4-1600" and "ddr_cntrl" are the non-blocking DDR controller's
#: organization (rtl/include/ddr_cntrl/dram_pkg.vh on FA26_integration:
#: 4Gb x8, 2 ranks of 4 x 4 banks, 15 row and 10 column bits) at
#: DDR4-1600J, as JEDEC gives it and with the controller's own values:
#:   nCL 11    the read FSM waits tRL = tAL + tCL = 1 + 10 (fsm_module.sv);
#:             Ramulator has no additive latency, so AL is folded into CL.
#:   nCWL 10   the write path counts tCWL (nb_wdata_queue.sv); DDR4-1600 is 9
#:   nRAS 45, nRC 55 (spec 28, 38): legal, rows stay open longer
#:   nRFC 172  spec 208 (260 ns for 4Gb): too short
#:   nREFI 250 shortened in the RTL to watch refreshes (spec 6240): with
#:             nRFC 172 the device refreshes most of the time
DRAM_CONFIGS: Dict[str, Dict] = {
    "rtl": dict(org="DDR4_8Gb_x16", preset="DDR4_3200W", ranks=2, channels=1,
                mapper="RoBaRaCoCh", row_policy="Open", read_buffer=256,
                write_buffer=64, coalesce_bytes=64),
    "ddr4-1600": dict(org="DDR4_4Gb_x8", preset="DDR4_1600J", ranks=2, channels=1,
                      mapper="RoBaRaCoCh", row_policy="Open", read_buffer=32,
                      write_buffer=32, coalesce_bytes=0),
}
DRAM_CONFIGS["ddr_cntrl"] = dict(DRAM_CONFIGS["ddr4-1600"], nCL=11, nCWL=10, nRAS=45,
                                 nRC=55, nRFC=172, nREFI=250)

#: Fields that are not Ramulator's DDR4 values.
_FIELDS = ("org", "preset", "ranks", "channels", "mapper", "row_policy",
           "read_buffer", "write_buffer", "coalesce_bytes")


def dram_params(config: str = "rtl", **overrides) -> Dict:
    """DRAM_CONFIGS[config] with `overrides` on top."""
    if config not in DRAM_CONFIGS:
        raise ValueError("DRAM config: one of %s, not %r" % (", ".join(DRAM_CONFIGS), config))
    return {**DRAM_CONFIGS[config], **overrides}


def ramulator_config(config: str = "rtl", **overrides) -> str:
    """The Ramulator 2 configuration for dram_params(config, **overrides),
    as the YAML its C++ side reads."""
    p = dram_params(config, **overrides)
    dram_overrides = {k: v for k, v in p.items() if k not in _FIELDS}
    path = os.path.join(RAMULATOR_DIR, "python")
    if path not in sys.path:
        sys.path.insert(0, path)
    import ramulator
    from ramulator.export import dict_to_yaml

    def ctrl():
        return ramulator.controller.GenericDDR(
            dram=ramulator.dram.DDR4(org_preset=p["org"], timing_preset=p["preset"],
                                     rank=int(p["ranks"]), **dram_overrides),
            scheduler=ramulator.scheduler.FRFCFS(),
            refresh_manager=ramulator.refresh_manager.AllBank(),
            row_policy=getattr(ramulator.row_policy, p["row_policy"])(),
            addr_mapper=getattr(ramulator.addr_mapper, p["mapper"])(),
            read_buffer_size=int(p["read_buffer"]),
            write_buffer_size=int(p["write_buffer"]))

    mem = ramulator.memory_system.GenericDRAM(
        clock_ratio=1, controllers=[ctrl() for _ in range(int(p["channels"]))],
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave())
    return dict_to_yaml({"frontend": ramulator.frontend.External(clock_ratio=1).to_config(),
                         "memory_system": mem.to_config()})


class RamulatorDRAM(Clocked):
    """The DRAM channel the DMA backends and cache ports share, timed by
    Ramulator 2; see the module docstring. `config` is a DRAM_CONFIGS name
    and `overrides` change any of its fields or Ramulator's DDR4 values;
    `core_mhz` sets how many memory cycles each core cycle runs (tCK and
    core_mhz give the ratio: 2 for DDR4-3200 at 800 MHz)."""

    #: Masters check this to wait for on_done instead of counting dram_latency.
    models_latency = True

    def __init__(self, config: str = "rtl", *, core_mhz: float = CORE_MHZ, **overrides):
        super().__init__()
        lib = _load()
        if lib is None:
            raise RuntimeError("Ramulator is not built: %s" % _load_error)
        self._lib = lib
        self.config = config
        self.params = dram_params(config, **overrides)
        self._h = lib.atr_create(ramulator_config(config, **overrides).encode())
        if not self._h:
            raise ValueError("Ramulator config: %s" % lib.atr_last_error().decode())
        self.tx_bytes = lib.atr_tx_bytes(self._h)
        self.tck_ns = lib.atr_tck_ns(self._h)
        #: Memory cycles per core cycle.
        self.ratio = Fraction(1000 / float(core_mhz) / self.tck_ns).limit_denominator(1000)
        self.last_issue_tick = -1
        self._cycle = -1          # last core cycle Ramulator has run
        self._next_tag = 0
        self._pending: Dict[int, Callable[[int], None]] = {}
        #: coalesce_bytes > 0: block address -> its outstanding read's waiters.
        self.coalesce_bytes = int(self.params["coalesce_bytes"])
        if self.coalesce_bytes > self.tx_bytes:
            raise ValueError("coalesce_bytes %d is more than a %d-byte DRAM transaction"
                             % (self.coalesce_bytes, self.tx_bytes))
        self._reading: Dict[int, list] = {}
        self._buf = (ctypes.c_int64 * 256)()
        self.stats = dict(reads=0, writes=0, refused=0, coalesced=0)

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.atr_destroy(h)

    # -- masters' side ------------------------------------------------------------------
    def can_issue(self, tick: int) -> bool:
        return int(tick) != self.last_issue_tick

    def reserve(self, tick: int) -> None:
        self.last_issue_tick = int(tick)

    def send(self, tick: int, addr: int, nbytes: int, is_write: bool,
             on_done: Callable[[int], None]) -> bool:
        """Offer a burst in cycle `tick`; False if the controller's queue is
        full. on_done(c) runs in the cycle c it completes."""
        if not 0 < nbytes <= self.tx_bytes:
            raise ValueError("a %d-byte burst does not fit a %d-byte DRAM transaction"
                             % (nbytes, self.tx_bytes))
        self._run_to(int(tick) - 1)
        block = None
        if self.coalesce_bytes and not is_write:
            block = int(addr) & ~(self.coalesce_bytes - 1)
            if block in self._reading:
                self._reading[block].append(on_done)
                self.stats["coalesced"] += 1
                return True
            addr, nbytes = block, self.coalesce_bytes
        tag = self._next_tag
        ok = self._lib.atr_send(self._h, int(bool(is_write)), int(addr), int(nbytes), tag)
        if ok < 0:
            raise RuntimeError(self._lib.atr_last_error().decode())
        if not ok:
            self.stats["refused"] += 1
            return False
        self._next_tag += 1
        if block is not None:
            waiters = self._reading[block] = [on_done]
            on_done = lambda cycle, block=block, waiters=waiters: self._read_done(block, cycle)
        self._pending[tag] = on_done
        self.stats["writes" if is_write else "reads"] += 1
        self.wake()
        return True

    def _read_done(self, block: int, cycle: int) -> None:
        for on_done in self._reading.pop(block):
            on_done(cycle)

    # -- the memory clock ---------------------------------------------------------------
    def _mem_cycles(self, first: int, last: int) -> int:
        """Memory cycles run in core cycles first..last."""
        r = self.ratio
        return int((last + 1) * r) - int(first * r)

    def _run_to(self, cycle: int) -> None:
        """Run Ramulator through core cycle `cycle`; completions land in it.
        While asleep nothing is in flight, so the catch-up completes nothing:
        it only keeps refresh where it would be had every cycle ticked."""
        if cycle <= self._cycle:
            return
        self._lib.atr_tick(self._h, self._mem_cycles(self._cycle + 1, cycle))
        self._cycle = cycle
        while True:
            n = self._lib.atr_drain(self._h, self._buf, len(self._buf))
            for i in range(n):
                self._pending.pop(self._buf[i])(cycle)
            if n < len(self._buf):
                break

    def tick(self, time=None) -> None:
        cycle = self._consume_tick(time)
        if cycle is not None:
            self._run_to(cycle)

    def next_wake(self, now: int) -> Optional[int]:
        return now + 1 if self._pending else None

    @property
    def idle(self) -> bool:
        return not self._pending

    def ramulator_stats(self, through: Optional[int] = None) -> str:
        """Ramulator's own statistics (row hits, latencies, ...) as YAML.
        Asleep at the end of a run, it has not counted the last idle
        cycles: `through` (the run's last cycle) runs them first, for the
        statistics that depend on time (cycles, throughput, queue averages)."""
        if through is not None:
            self._run_to(int(through))
        return self._lib.atr_stats(self._h).decode()

    def controller_stats(self, through: Optional[int] = None) -> Dict[str, float]:
        """The controllers' counters, summed over channels: row_hits,
        row_misses, row_conflicts, num_read_reqs_served, read_latency
        (memory cycles, summed over reads), ..."""
        out: Dict[str, float] = {}
        section = None
        for line in self.ramulator_stats(through).splitlines():
            key, _, val = line.strip().partition(":")
            if key == "impl":
                section = val.strip()
            elif section == "GenericDDR" and val.strip():
                try:
                    out[key] = out.get(key, 0) + float(val)
                except ValueError:
                    pass
        return out

    def summary(self, through: Optional[int] = None) -> str:
        s = self.controller_stats(through)
        reads = s.get("num_read_reqs_served", 0)
        rows = sum(s.get(k, 0) for k in ("row_hits", "row_misses", "row_conflicts"))
        lines = ["DRAM (Ramulator 2, %s: %s %s, tCK %.3g ns, %s memory cycles a core cycle)"
                 % (self.config, self.params["preset"], self.params["org"], self.tck_ns,
                    self.ratio)]
        lines.append("  %d reads, %d writes, %d refused for a full queue"
                     % (self.stats["reads"], self.stats["writes"], self.stats["refused"]))
        if self.stats["coalesced"]:
            lines.append("  %d reads merged into a read of the same %d-byte block"
                         % (self.stats["coalesced"], self.coalesce_bytes))
        if rows:
            lines.append("  row hits %.1f%%, misses %.1f%%, conflicts %.1f%%"
                         % tuple(100 * s.get(k, 0) / rows
                                 for k in ("row_hits", "row_misses", "row_conflicts")))
        if reads:
            lines.append("  average read latency %.1f memory cycles"
                         % (s.get("read_latency", 0) / reads))
        return "\n".join(lines)
