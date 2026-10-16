"""Activity probe: which blocks of the chip are working, cycle by cycle.

A read-only observer of AtallaPlatform, ticked last in every cycle like the
performance monitor. Each cycle it puts every block of the chip in one of
three states and records them as intervals, so a long run stays small:

    IDLE      nothing to do
    ACTIVE    holds work and is moving it
    STALLED   holds work but is held up (a bank conflict, a busy DRAM
              channel, a full writeback)

    plat = build_atalla_platform(instr, data, activity=True)
    plat.run_until_done()
    plat.activity.to_json("trace.json")      # tools/activity_view.py draws it

The blocks follow the chip's top-level diagram -- fetch, decode 1 and 2,
scalar execute, the vector core, the systolic array, memory -- drawn as the
model is built (four scratchpads, `lane_count` lanes, the transpose unit).

To add a block: give it an entry in `_blocks` (an id, a label, the panel it
is drawn in, and what active and stalled mean) and a line in `_sample` that
returns its state. Read the platform; never call anything that changes it.
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from base.clocked_object import Clocked

IDLE, ACTIVE, STALLED = 0, 1, 2
STATE_NAMES = ("idle", "active", "stalled")


@dataclass
class Block:
    """One drawn block: what it is, where it goes, what its states mean."""
    id: str
    label: str
    panel: str
    active: str                      # what ACTIVE means for this block
    stalled: str = ""                # what STALLED means, if it can stall


class ActivityProbe(Clocked):
    """Records each block's state every cycle; see the module docstring."""

    def __init__(self, platform):
        super().__init__()
        self.p = platform
        self._tick = -1
        self.cycles = 0
        self.blocks: List[Block] = self._blocks()
        #: block id -> [[first cycle, last cycle, state], ...], idle not stored
        self.intervals: Dict[str, List[List[int]]] = {b.id: [] for b in self.blocks}
        self._state: Dict[str, int] = {b.id: IDLE for b in self.blocks}
        #: The performance monitor's cycle bucket, as [[first, last, name], ...]
        self.buckets: List[list] = []
        # Counters whose change between cycles says something happened.
        self._prev = {"issued": 0, "to_d1d2": 0, "mul_ops": 0, "pc": None}
        self._prev_stalls = [0] * len(platform.backends)
        self._prev_contention = {n: 0 for n in platform.ports}

    # -- the blocks ---------------------------------------------------------------------
    def _blocks(self) -> List[Block]:
        p = self.p
        lanes = len(p.vc.datapath.lanes)
        array = "MEISSA" if hasattr(p.sa, "_is_drained") else "TPU"
        b = [
            Block("fetch", "PC + BTB", "Fetch", "a packet moved into IF/D1",
                  "waiting for the iCache"),
            Block("icache", "iCache", "Fetch", "filling a line"),
            Block("decode1", "Decode 0-3 + xbar", "Decode 1", "IF/D1 holds a packet"),
            Block("d2_scalar", "Scalar decode", "Decode 2",
                  "a packet with scalar ops issued", "a packet with scalar ops held"),
            Block("d2_vector", "Vector decode", "Decode 2",
                  "a packet with vector ops issued", "a packet with vector ops held"),
            Block("d2_sdma", "SDMA decode", "Decode 2",
                  "a packet with a DMA op issued", "a packet with a DMA op held"),
            Block("srf", "Scalar reg file", "Decode 2", "operands read",
                  "a bank conflict"),
            Block("vrf", "Vector reg file", "Decode 2", "operands read",
                  "a bank conflict"),
            Block("mrf", "Mask reg file", "Decode 2", "operands read", "a bank conflict"),
            Block("ex1", "ALU + branch (EX1)", "Scalar execute", "an op in EX1",
                  "a result waiting for writeback"),
            Block("ex2", "Div / convert / sqrt (EX2)", "Scalar execute", "an op in flight"),
            Block("ex3", "BF16 FP ALU (EX3)", "Scalar execute", "an op in flight"),
            Block("ex4", "Multiply (EX4)", "Scalar execute", "an op in flight"),
            Block("lsu", "Load/store (EX5)", "Scalar execute", "requests queued",
                  "only waiting for cache misses"),
            Block("dcache", "dCache", "Scalar execute", "a lookup or a fill"),
        ]
        b += [Block("lane%d" % i, "Lane %d" % i, "Vector core",
                    "a functional unit holds work") for i in range(lanes)]
        b += [
            Block("gsau", "GSAU", "Vector core", "rows to or from the array"),
            Block("transpose", "Transpose", "Vector core", "a row or column in flight"),
            Block("mts", "Move to scalar", "Vector core", "",
                  "a result waiting for writeback"),
        ]
        b += [Block("vlsu%d" % i, "Vec LD/ST %d" % i, "Vector core",
                    "a load or store in flight") for i in range(len(p.vc.vls_units))]
        b += [
            Block("array", "PE grid (%s %dx%d)" % (array, p.sa.size, p.sa.size),
                  "Systolic array", "multiplies this cycle"),
            Block("array_out", "Output buffer", "Systolic array", "results in flight"),
        ]
        b += [Block("spad%d" % i, "Scratchpad %d" % i, "Memory",
                    "requests queued or in flight") for i in range(len(p.spad.tiles))]
        b += [Block("sdma%d" % i, "DMA %d" % i, "Memory", "a transfer running",
                    "its burst lost the DRAM channel") for i in range(len(p.backends))]
        b += [Block("%s_port" % n, "%s port" % ("iCache" if n == "icache" else "dCache"),
                    "Memory", "bursts queued or in flight", "lost the DRAM channel")
              for n in p.ports]
        b += [
            Block("channel", "DRAM channel", "Memory", "a burst launched this cycle"),
            Block("dram", "DRAM (fixed latency)", "Memory", "bursts outstanding"),
        ]
        return b

    # -- one cycle ----------------------------------------------------------------------
    def tick(self, time: Optional[float] = None) -> None:
        cycle = self._consume_tick(time, attr_name="_tick")
        if cycle is None:
            return
        core = self.p.core
        if core.done_at is not None and cycle > core.done_at:
            return
        self.cycles = cycle + 1
        for bid, state in self._sample(cycle).items():
            self._record(bid, state, cycle)
        perf = self.p.perf
        if perf is not None and perf.timeline:
            name = perf.timeline[-1]
            if self.buckets and self.buckets[-1][2] == name and self.buckets[-1][1] == cycle - 1:
                self.buckets[-1][1] = cycle
            else:
                self.buckets.append([cycle, cycle, name])

    def _record(self, bid: str, state: int, cycle: int) -> None:
        if state == IDLE:
            return
        runs = self.intervals[bid]
        if runs and runs[-1][2] == state and runs[-1][1] == cycle - 1:
            runs[-1][1] = cycle
        else:
            runs.append([cycle, cycle, state])

    def _sample(self, cycle: int) -> Dict[str, int]:
        p, core, s = self.p, self.p.core, {}
        prev = self._prev
        d2 = core.decode2

        # Front end.
        moved = len(core.issued_to_d1d2) > prev["to_d1d2"]
        prev["to_d1d2"] = len(core.issued_to_d1d2)
        filling = core.icache.state != 0
        s["icache"] = ACTIVE if filling else IDLE
        s["fetch"] = ACTIVE if moved else STALLED if filling else IDLE
        s["decode1"] = ACTIVE if core.ifd1.valid else IDLE

        # Decode 2: the packet in D1/D2, issued this cycle or held.
        issued = len(core.issued) > prev["issued"]
        prev["issued"] = len(core.issued)
        latch = core.d1d2
        for bid, words in (("d2_scalar", latch.scalar), ("d2_vector", latch.vector),
                           ("d2_sdma", latch.sdma)):
            if not (latch.valid and any(words)):
                s[bid] = IDLE
            else:
                s[bid] = ACTIVE if issued else STALLED
        for bid, rf in (("srf", d2.srf), ("vrf", d2.vrf), ("mrf", d2.mrf)):
            reading = latch.valid and rf._reads
            s[bid] = STALLED if rf.state == "CONFLICT" else ACTIVE if reading else IDLE

        # Scalar execute.
        ex1 = core.ex1
        s["ex1"] = (STALLED if ex1.held is not None
                    else ACTIVE if getattr(ex1, "_in", None) is not None else IDLE)
        for name in ("ex2", "ex3", "ex4"):
            s[name] = ACTIVE if getattr(core, name).busy else IDLE
        lsu = core.lsu
        if lsu.idle:
            s["lsu"] = IDLE
        elif lsu.queue or lsu.in_lookup or lsu.wb:
            s["lsu"] = ACTIVE
        else:
            s["lsu"] = STALLED
        s["dcache"] = IDLE if core.dcache.idle() else ACTIVE

        # Vector core.
        vc = p.vc
        vs = core.vector
        for i, lane in enumerate(vc.datapath.lanes):
            s["lane%d" % i] = IDLE if lane.is_idle() else ACTIVE
        g = vc.gsau
        s["gsau"] = (ACTIVE if g.has_pending() or not g.rd_queue.is_empty()
                     or not g.writebacks.is_empty() else IDLE)
        busy_tp = vc.transpose.busy or (vs is not None and (vs._tp_reqs or vs._tp_vds))
        s["transpose"] = ACTIVE if busy_tp else IDLE
        s["mts"] = STALLED if vs is not None and vs.mts.held is not None else IDLE
        for i, v in enumerate(vc.vls_units):
            pending = not (v.issue_q.is_empty() and v.req_q.is_empty() and v.rsp_q.is_empty()
                           and v.wb_q.is_empty()) or v.outstanding_loads()
            s["vlsu%d" % i] = ACTIVE if pending else IDLE

        # Systolic array.
        sa = p.sa
        mul = sa.metrics.get("mul_ops", 0)
        s["array"] = ACTIVE if mul > prev["mul_ops"] else IDLE
        prev["mul_ops"] = mul
        if hasattr(sa, "_is_drained"):                  # MEISSA
            out = not sa._is_drained()
        else:
            out = any(row is not None for row in sa._output_pipe)
        s["array_out"] = ACTIVE if out else IDLE

        # Memory.
        for i, pad in enumerate(p.spad.tiles):
            s["spad%d" % i] = IDLE if pad.idle() else ACTIVE
        sdma_busy = vs.sdma_busy if vs is not None else [False] * len(p.backends)
        for i, be in enumerate(p.backends):
            lost = be.total_backend_stalls > self._prev_stalls[i]
            self._prev_stalls[i] = be.total_backend_stalls
            s["sdma%d" % i] = STALLED if lost else ACTIVE if sdma_busy[i] else IDLE
        for n, port in p.ports.items():
            lost = port.stats["contention_cycles"] > self._prev_contention[n]
            self._prev_contention[n] = port.stats["contention_cycles"]
            s["%s_port" % n] = (STALLED if lost
                                else ACTIVE if port._launch_q or port._due else IDLE)
        channel = p.backends[0].shared_burst_channel if p.backends else None
        s["channel"] = ACTIVE if channel is not None and channel.last_issue_tick == cycle else IDLE
        outstanding = (any(not be._dram_pending.is_empty() for be in p.backends)
                       or any(port._due for port in p.ports.values()))
        s["dram"] = ACTIVE if outstanding else IDLE
        return s

    # -- results ------------------------------------------------------------------------
    def counts(self, bid: str) -> Dict[str, int]:
        """Cycles a block spent active and stalled."""
        out = {"active": 0, "stalled": 0}
        for first, last, state in self.intervals[bid]:
            out[STATE_NAMES[state]] += last - first + 1
        return out

    def trace(self, meta: Optional[dict] = None) -> dict:
        """Everything the viewer needs, as plain data."""
        return {
            "version": 1,
            "cycles": self.cycles,
            "meta": dict(meta or {}),
            "states": list(STATE_NAMES),
            "blocks": [dict(id=b.id, label=b.label, panel=b.panel, active=b.active,
                            stalled=b.stalled, intervals=self.intervals[b.id])
                       for b in self.blocks],
            "buckets": self.buckets,
        }

    def to_json(self, path: str, meta: Optional[dict] = None) -> None:
        with open(path, "w") as f:
            json.dump(self.trace(meta), f, separators=(",", ":"))
