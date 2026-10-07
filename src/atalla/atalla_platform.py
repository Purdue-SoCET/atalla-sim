"""The Atalla platform: the scheduler driving the vector core, the systolic
array, the scratchpads and DRAM.

The same parts as TPUPlatform (atalla/sysarr_tpu_system.py), put together
the way rtl/modules/system.sv puts them: the vector core's units, four
scratchpad pads with their DMA backends, DRAM, the VLSU-to-scratchpad
bridges, and the GSAU-to-systolic-array bridge. What TPUPlatform leaves to a
test harness -- deciding what each unit does and when -- the scheduler core
(scheduler/core.py) does here: it fetches and decodes the program, and in
each packet's EX cycle calls into the units (scheduler/vector.py):

    lane ops        vc.datapath.enqueue(...)
    gemm.vv, lw.vi  vc.gsau.issue(...)
    vreg.ld/st      vc.vls_units[sid].issue(...)
    scpad.ld/st     backends[sid].driver_to_backend_start_load/store(...)

and takes their results back through writeback. The units keep their own
timing; the scheduler only routes, arbitrates and waits on their readiness.

    SchedulerCore -> VectorCore units -> VLSU bridges -> scratchpad -> backends -+
                  \\-> GSAU -> array bridge -> systolic array (MEISSA or TPU)       |
                  \\-> icache, dcache --------------------------------> bus ports -+-> DRAM

Memory is sim_ram_rr's arrangement: the icache, the dcache and the four
scratchpad backends are six masters on one DRAM channel, one burst launch a
cycle, round robin (memory/bus.py for the caches' ports). The icache
starts cold.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from atalla.sysarr_tpu_system import (
    PHASE_BACKEND, PHASE_CORE, PHASE_HARNESS, PHASE_SPAD, PHASE_SYSARR, PHASE_VLS,
    GSAUTPUBridge, RoundRobinBackendTicker, build_tpu_platform)
from atalla.perf_monitor import PerfMonitor
from base.sched import CompositeClocked
from memory.backend import Backend
from memory.bus import BusMaster
from memory.dram import DRAM
from scheduler.core import SchedulerCore, load_program_text

MASK32 = 0xFFFFFFFF


def c_entry_sregs(data: Optional[Dict[int, int]]) -> Dict[int, int]:
    """The scalar registers the functional sim's C-kernel loader sets before
    a compiled program starts (core/kernel_services.run_emulator_from_in_path):
    the stack pointer x2 a page above the highest data address, and x33 a
    page above that. Hand-written assembly kernels don't need them."""
    top = max(data) if data else 0
    stack = ((top + 0x1000) & ~0xFFF) + 0x1000
    return {2: stack, 33: stack + 0x1000}


class DramWords:
    """The data cache's word memory, over the platform's DRAM (little-endian
    32-bit words)."""

    def __init__(self, dram: DRAM):
        self.dram = dram

    def read(self, addr: int) -> int:
        return int.from_bytes(self.dram.read(int(addr) & ~3, 4), "little")

    def write(self, addr: int, value: int) -> None:
        self.dram.write(int(addr) & ~3, (int(value) & MASK32).to_bytes(4, "little"))


class ArrayValueBridge(GSAUTPUBridge):
    """The GSAU-to-systolic-array bridge, returning the array's results as
    values. The harnesses' bridge packs them as FP16 bit patterns, because
    their vector registers hold raw 16-bit lanes; the scheduler's hold BF16
    values, and the array already rounds its outputs to BF16."""

    def _pack_rsp(self, out_row, meta):
        vec = [0.0] * self.vc.vector_len
        for i, val in enumerate(out_row[: self.size]):
            vec[i] = float(val)
        return {"vdata": vec, "meta": dict(meta), "dtype": meta.get("dtype")}


@dataclass
class AtallaPlatform:
    core: SchedulerCore
    vc: Any                          # VectorCore
    spad: Any                        # Scratchpad
    sa: Any                          # SystolicArrayMEISSA or SystolicArrayTPU
    dram: DRAM
    backends: List[Backend]
    vls_bridges: List[Any]
    sysarr_bridge: GSAUTPUBridge
    ports: Dict[str, BusMaster]
    root: CompositeClocked
    perf: Optional[PerfMonitor] = None
    cycle: int = 0

    def tick(self) -> None:
        self.root.tick(float(self.cycle))
        self.cycle += 1

    def run_until_done(self, limit: int = 1_000_000) -> int:
        start = self.cycle
        while not self.core.done:
            if self.cycle - start >= limit:
                raise RuntimeError("no halt after %d cycles (pc %#x)" % (limit, self.core.pc))
            self.tick()
        return self.cycle - start


def build_atalla_platform(program: Dict[int, int], data: Optional[Dict[int, int]] = None, *,
                          systolic_array: str = "meissa", lane_count: int = 16,
                          dram_latency: int = 6, burst_bytes: int = 32,
                          warm_icache: bool = False, perf: bool = True,
                          perf_timeline: bool = False,
                          init_sregs: Optional[Dict[int, int]] = None,
                          **core_kw) -> AtallaPlatform:
    """`program` and `data` as load_program_text returns them; `data` words
    go into DRAM. `systolic_array` is "meissa" (the RTL's) or "tpu"; extra
    keywords go to SchedulerCore. `perf` attaches a PerfMonitor
    (atalla/perf_monitor.py) as platform.perf; `perf_timeline` makes it keep
    the per-cycle timeline too. `init_sregs` presets scalar registers, as a
    loader would (c_entry_sregs for compiled C)."""
    parts = build_tpu_platform(dtype="bf16", lane_count=lane_count,
                               systolic_array=systolic_array,
                               backend_dram_latency=dram_latency,
                               backend_dram_burst_bytes=burst_bytes)
    vc, dram, backends = parts.vc, parts.dram, parts.backends
    for addr, word in (data or {}).items():
        dram.write(int(addr), (int(word) & MASK32).to_bytes(4, "little"))

    channel = backends[0].shared_burst_channel
    ports = {name: BusMaster(name, channel, dram_latency=dram_latency, burst_bytes=burst_bytes)
             for name in ("icache", "dcache")}
    core = SchedulerCore(program, vector_core=vc, backends=backends,
                         memory=DramWords(dram), icache_port=ports["icache"],
                         dcache_port=ports["dcache"], **core_kw)
    if warm_icache:
        core.warm_icache()
    for r, v in (init_sregs or {}).items():
        core.decode2.srf.storage.write(int(r), int(v) & MASK32)
    sysarr_bridge = ArrayValueBridge(vc, parts.sa)

    # build_tpu_platform's tree, with the scheduler core in the vector core's
    # place: it ticks the vector core's units itself.
    root = CompositeClocked("atalla_platform")
    root.add_child(core, phase=PHASE_CORE)
    for bridge in parts.vls_bridges:
        root.add_child(bridge, phase=PHASE_VLS)
    root.add_child(sysarr_bridge, phase=PHASE_SYSARR)
    root.add_child(parts.spad, phase=PHASE_SPAD)
    # sim_ram_rr's order: icache, dcache, scratchpads 0-3.
    root.add_child(RoundRobinBackendTicker([ports["icache"], ports["dcache"], *backends]),
                   phase=PHASE_BACKEND)
    plat = AtallaPlatform(core=core, vc=vc, spad=parts.spad, sa=parts.sa, dram=dram,
                          backends=backends, vls_bridges=parts.vls_bridges,
                          sysarr_bridge=sysarr_bridge, ports=ports, root=root)
    if perf:
        # Last in every cycle: it observes and changes nothing.
        plat.perf = PerfMonitor(plat, timeline=perf_timeline)
        root.add_child(plat.perf, phase=PHASE_HARNESS)
    return plat


def build_kernel_platform(text: str, **kw) -> AtallaPlatform:
    """A platform loaded with an assembled program in the functional sim's
    `.in` format (what its kernels/build_*.py scripts write)."""
    instr, data = load_program_text(text)
    return build_atalla_platform(instr, data, **kw)
