"""A platform driven by the scheduler model instead of a harness.

build_tpu_platform (atalla/sysarr_tpu_system.py) assembles the vector core,
the four scratchpad pads with their backends, DRAM, the VLSU-to-frontend
bridges and the systolic array, and ticks the vector core in its harness
mode. Here the same parts are put together around a SchedulerCore, which
takes the vector core's place in the tick order and drives its units
itself (scheduler/vector.py):

    SchedulerCore -> VectorCore units -> VLSU bridges -> scratchpad -> backends -> DRAM
                  \\-> GSAU -> systolic array

The scalar data cache's memory is the same DRAM, as the dcache and the
scratchpad backends share sim_ram_rr in the RTL.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from atalla.sysarr_tpu_system import (
    PHASE_BACKEND, PHASE_CORE, PHASE_SPAD, PHASE_SYSARR, PHASE_VLS,
    RoundRobinBackendTicker, build_tpu_platform)
from base.sched import CompositeClocked
from memory.dram import DRAM
from scheduler.core import SchedulerCore

MASK32 = 0xFFFFFFFF


class DramWords:
    """The data cache's word memory, over the platform's DRAM (little-endian
    32-bit words)."""

    def __init__(self, dram: DRAM):
        self.dram = dram

    def read(self, addr: int) -> int:
        return int.from_bytes(self.dram.read(int(addr) & ~3, 4), "little")

    def write(self, addr: int, value: int) -> None:
        self.dram.write(int(addr) & ~3, (int(value) & MASK32).to_bytes(4, "little"))


@dataclass
class SchedulerPlatform:
    core: SchedulerCore
    tpu: object                     # the TPUPlatform the parts came from
    root: CompositeClocked
    cycle: int = 0

    @property
    def vc(self):
        return self.tpu.vc

    @property
    def dram(self) -> DRAM:
        return self.tpu.dram

    @property
    def spad(self):
        return self.tpu.spad

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


def build_scheduler_platform(program: Dict[int, int], data: Optional[Dict[int, int]] = None, *,
                             dram_latency: int = 6, lane_count: int = 4,
                             warm_icache: bool = True, **core_kw) -> SchedulerPlatform:
    """`program` and `data` as load_program_text returns them; `data` words
    go into DRAM, which both the data cache and the scratchpad DMA see."""
    tpu = build_tpu_platform(dtype="bf16", lane_count=lane_count,
                             backend_dram_latency=dram_latency)
    for addr, word in (data or {}).items():
        tpu.dram.write(int(addr), (int(word) & MASK32).to_bytes(4, "little"))
    core = SchedulerCore(program, vector_core=tpu.vc, backends=tpu.backends,
                         memory=DramWords(tpu.dram), **core_kw)
    if warm_icache:
        core.warm_icache()
    root = CompositeClocked("scheduler_platform")
    root.add_child(core, phase=PHASE_CORE)
    for bridge in tpu.vls_bridges:
        root.add_child(bridge, phase=PHASE_VLS)
    root.add_child(tpu.sysarr_bridge, phase=PHASE_SYSARR)
    root.add_child(tpu.spad, phase=PHASE_SPAD)
    if tpu.backends:
        root.add_child(RoundRobinBackendTicker(tpu.backends), phase=PHASE_BACKEND)
    return SchedulerPlatform(core, tpu, root)


def build_kernel_platform(text: str, **kw) -> SchedulerPlatform:
    """A platform loaded with an assembled program in the functional sim's
    `.in` format (what its kernels/build_*.py scripts write)."""
    from scheduler.core import load_program_text
    instr, data = load_program_text(text)
    return build_scheduler_platform(instr, data, **kw)
