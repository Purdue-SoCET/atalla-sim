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

The default configuration is the RTL DDR controller's
(rtl/include/ddr_cntrl/dram_pkg.vh): DDR4 4Gb x8 parts on a 64-bit
channel, 2 ranks of 4 bank groups of 4 banks, 15 row and 10 column bits,
CL-RCD-RP 10-10-10 (DDR4-1600J, an 800 MHz memory clock: one memory cycle
per 800 MHz core cycle). The address mapping is Ramulator's RoBaRaCoCh,
not the RTL address_mapper's {rank, row, bank, BG[1], col[9:3], BG[0],
col[2:0]}, which Ramulator has no mapper for.

Every burst is one DDR4 transaction of 64 bytes (BL8 on 64 bits), however
few bytes it carries: an 8-byte burst costs as much data-bus time as a
64-byte one.

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


def ramulator_config(*, org: str = "DDR4_4Gb_x8", timing: str = "DDR4_1600J",
                     ranks: int = 2, channels: int = 1, mapper: str = "RoBaRaCoCh",
                     row_policy: str = "Open", **dram_overrides) -> str:
    """A Ramulator 2 configuration, as the YAML its C++ side reads. The
    keywords pick Ramulator's DDR4 presets and components by name;
    `dram_overrides` go to its DDR4 object (nCL=11, ...)."""
    path = os.path.join(RAMULATOR_DIR, "python")
    if path not in sys.path:
        sys.path.insert(0, path)
    import ramulator
    from ramulator.export import dict_to_yaml

    def ctrl():
        return ramulator.controller.GenericDDR(
            dram=ramulator.dram.DDR4(org_preset=org, timing_preset=timing,
                                     rank=int(ranks), **dram_overrides),
            scheduler=ramulator.scheduler.FRFCFS(),
            refresh_manager=ramulator.refresh_manager.AllBank(),
            row_policy=getattr(ramulator.row_policy, row_policy)(),
            addr_mapper=getattr(ramulator.addr_mapper, mapper)())

    mem = ramulator.memory_system.GenericDRAM(
        clock_ratio=1, controllers=[ctrl() for _ in range(int(channels))],
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave())
    return dict_to_yaml({"frontend": ramulator.frontend.External(clock_ratio=1).to_config(),
                         "memory_system": mem.to_config()})


class RamulatorDRAM(Clocked):
    """The DRAM channel the DMA backends and cache ports share, timed by
    Ramulator 2; see the module docstring. `config` is ramulator_config()'s
    YAML (its defaults when None); `core_mhz` sets how many memory cycles
    each core cycle runs."""

    #: Masters check this to wait for on_done instead of counting dram_latency.
    models_latency = True

    def __init__(self, config: Optional[str] = None, *, core_mhz: float = CORE_MHZ):
        super().__init__()
        lib = _load()
        if lib is None:
            raise RuntimeError("Ramulator is not built: %s" % _load_error)
        self._lib = lib
        self._h = lib.atr_create((config or ramulator_config()).encode())
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
        self._buf = (ctypes.c_int64 * 256)()
        self.stats = dict(reads=0, writes=0, refused=0)

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
        tag = self._next_tag
        ok = self._lib.atr_send(self._h, int(bool(is_write)), int(addr), int(nbytes), tag)
        if ok < 0:
            raise RuntimeError(self._lib.atr_last_error().decode())
        if not ok:
            self.stats["refused"] += 1
            return False
        self._next_tag += 1
        self._pending[tag] = on_done
        self.stats["writes" if is_write else "reads"] += 1
        self.wake()
        return True

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
        lines = ["DRAM (Ramulator 2, tCK %.3g ns, %s memory cycles a core cycle)"
                 % (self.tck_ns, self.ratio)]
        lines.append("  %d reads, %d writes, %d refused for a full queue"
                     % (self.stats["reads"], self.stats["writes"], self.stats["refused"]))
        if rows:
            lines.append("  row hits %.1f%%, misses %.1f%%, conflicts %.1f%%"
                         % tuple(100 * s.get(k, 0) / rows
                                 for k in ("row_hits", "row_misses", "row_conflicts")))
        if reads:
            lines.append("  average read latency %.1f memory cycles"
                         % (s.get("read_latency", 0) / reads))
        return "\n".join(lines)
