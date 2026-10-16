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

The blocks follow the chip's top-level diagram -- Fetch, Decode1, Decode2,
Execute, the Systolic Array, Memory -- down to its boxes: each decode slot,
each register-file bank, each lane, each scratchpad and its crossbars. Where
the model has more or other parts than the diagram (four scratchpads,
`lane_count` lanes, the transpose unit, separate EX2 and EX4), the blocks
follow the model.

To add a block: give it an entry in `_blocks` (an id, a label, the panel it
is drawn in, and what active and stalled mean) and a line in `_sample` that
returns its state. Read the platform; never call anything that changes it.
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from base.clocked_object import Clocked
from scheduler.decode1 import SCALAR_SLOTS as SLOTS, slot_word
from scheduler.isa import OP_W, OPCODES
from scheduler.semantics import is_control

IDLE, ACTIVE, STALLED = 0, 1, 2
STATE_NAMES = ("idle", "active", "stalled")

#: EX2 ops that are conversions; the rest of EX2 is divide, mod, rcp and sqrt.
CONVERSIONS = ("bfts.s", "stbf.s")


def _is_instruction(word: int) -> bool:
    """A real instruction in a packet slot (not empty, not a nop)."""
    if not word:
        return False
    return OPCODES.get(word & ((1 << OP_W) - 1), ("",))[0] != "nop.s"


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
        self._prev = {"issued": 0, "to_d1d2": 0, "mul_ops": 0, "act_rows": 0}
        self._prev_stalls = [0] * len(platform.backends)
        self._prev_contention = {n: 0 for n in platform.ports}

    # -- the blocks ---------------------------------------------------------------------
    def _blocks(self) -> List[Block]:
        p = self.p
        meissa = hasattr(p.sa, "_is_drained")
        b = [
            Block("bpu", "Branch Prediction Unit", "Fetch",
                  "a prediction used, or the BTB written"),
            Block("pc", "PC", "Fetch", "a packet moved into IF/D1", "waiting for the iCache"),
            Block("icache", "iCache", "Fetch", "filling a line"),
        ]
        b += [Block("dec1_s%d" % i, "Decode %d" % i, "Decode1",
                    "slot %d of the packet in IF/D1 holds an instruction" % i)
              for i in range(SLOTS)]
        b += [Block("decode1_xbar", "Decode1 xbar", "Decode1", "IF/D1 holds a packet")]
        for kind, label, n in (("s", "ALU decode", SLOTS), ("v", "Vector decode", SLOTS),
                               ("d", "SDMA decode", SLOTS)):
            b += [Block("d2_%s%d" % (kind, i), "%s %d" % (label, i), "Decode2",
                        "slot %d holds an op and the packet issued" % i,
                        "slot %d holds an op and the packet is held" % i) for i in range(n)]
        for rf, name in (("srf", "Sclr"), ("vrf", "Vec")):
            regs = getattr(p.core.decode2, rf)
            b.append(Block("%s_read" % rf, "%s issue reg read" % name, "Decode2", "operands read"))
            b += [Block("%s_b%d" % (rf, k), "%s Reg File bank %d" % (name, k), "Decode2",
                        "a register in bank %d read" % k) for k in range(regs.banks)]
            b.append(Block("%s_oc" % rf, "%s operand collectors" % name, "Decode2",
                           "holding the operands of a conflicting packet",
                           "serving a bank conflict"))
        b.append(Block("mrf", "Mask reg file", "Decode2", "masks read", "a bank conflict"))
        b += [
            Block("ex1_alu", "Sclr ALU", "Execute", "an ALU op in EX1",
                  "its result waiting for writeback"),
            Block("ex1_br", "Branch", "Execute", "a branch or jump in EX1",
                  "its result waiting for writeback"),
            Block("ex2_div", "Divide / sqrt (EX2)", "Execute", "a divide, mod, rcp or sqrt"),
            Block("ex2_cvt", "int to bf16 (EX2)", "Execute", "a conversion"),
            Block("ex3", "FP ALU (EX3)", "Execute", "a BF16 add, sub, mul or compare"),
            Block("ex4", "Multiply (EX4)", "Execute", "an integer multiply"),
            Block("lsu", "SCLR LD/ST", "Execute", "requests queued",
                  "only waiting for cache misses"),
        ]
        b += [Block("lane%d" % i, "Lane %d" % i, "Execute", "a functional unit holds work")
              for i in range(len(p.vc.datapath.lanes))]
        b += [
            Block("gsau", "GSAU", "Execute", "rows to or from the array"),
            Block("transpose", "Transpose", "Execute", "a row or column in flight"),
            Block("mts", "Move to scalar", "Execute", "", "a result waiting for writeback"),
        ]
        b += [Block("vlsu%d" % i, "Vec LD/ST %d" % i, "Execute", "a load or store in flight")
              for i in range(len(p.vc.vls_units))]
        b += [Block("array_in", "input fifo", "Systolic Array", "a row queued for the array")]
        if not meissa:
            b += [Block("psum_in", "psum input fifo", "Systolic Array", "partial sums queued")]
        b += [
            Block("array", "PE grid (%s %dx%d)" % ("MEISSA" if meissa else "TPU",
                                                   p.sa.size, p.sa.size),
                  "Systolic Array", "multiplies this cycle"),
            Block("array_out", "psum output fifo", "Systolic Array", "results in flight"),
            Block("dcache", "dCache", "Memory", "a lookup or a fill"),
        ]
        for i in range(len(p.spad.tiles)):
            b += [Block("spad%d_w" % i, "Scpad %d write xbar" % i, "Memory", "writes queued"),
                  Block("spad%d" % i, "Scpad %d" % i, "Memory", "requests queued or in flight"),
                  Block("spad%d_r" % i, "Scpad %d read xbar" % i, "Memory", "reads queued or returning")]
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
            if bid in self.intervals:
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

        # Fetch and decode 1.
        moved = len(core.issued_to_d1d2) > prev["to_d1d2"]
        prev["to_d1d2"] = len(core.issued_to_d1d2)
        filling = core.icache.state != 0
        s["icache"] = ACTIVE if filling else IDLE
        s["pc"] = ACTIVE if moved else STALLED if filling else IDLE
        s["bpu"] = ACTIVE if (core.fetch.out_predict_taken or core.btb.in_update_en) else IDLE
        ifd1 = core.ifd1
        s["decode1_xbar"] = ACTIVE if ifd1.valid else IDLE
        for i in range(SLOTS):
            word = slot_word(ifd1.inst_packet, i) if ifd1.valid else 0
            s["dec1_s%d" % i] = ACTIVE if _is_instruction(word) else IDLE

        # Decode 2: the packet in D1/D2, issued this cycle or held.
        issued = len(core.issued) > prev["issued"]
        prev["issued"] = len(core.issued)
        latch = core.d1d2
        for kind, words in (("s", latch.scalar), ("v", latch.vector), ("d", latch.sdma)):
            for i, word in enumerate(words):
                if latch.valid and _is_instruction(word):
                    s["d2_%s%d" % (kind, i)] = ACTIVE if issued else STALLED
                else:
                    s["d2_%s%d" % (kind, i)] = IDLE
        for name in ("srf", "vrf", "mrf"):
            rf = getattr(d2, name)
            reads = list(rf._reads) if latch.valid else []
            conflict = rf.state == "CONFLICT"
            if name == "mrf":
                s["mrf"] = STALLED if conflict else ACTIVE if reads else IDLE
                continue
            s[name + "_read"] = ACTIVE if reads else IDLE
            for k in range(rf.banks):
                s["%s_b%d" % (name, k)] = ACTIVE if any(r % rf.banks == k for r in reads) else IDLE
            s[name + "_oc"] = STALLED if conflict else ACTIVE if rf.state == "DONE" else IDLE

        # Scalar execute.
        ex1 = core.ex1
        held = ex1.held[0] if ex1.held is not None else None
        live = held or getattr(ex1, "_in", None)
        branch = live is not None and is_control(live.op)
        state1 = STALLED if held is not None else ACTIVE if live is not None else IDLE
        s["ex1_br"] = state1 if branch else IDLE
        s["ex1_alu"] = IDLE if branch else state1
        ex2 = core.ex2
        convert = ex2.mnemonic in CONVERSIONS
        s["ex2_cvt"] = ACTIVE if ex2.busy and convert else IDLE
        s["ex2_div"] = ACTIVE if ex2.busy and not convert else IDLE
        s["ex3"] = ACTIVE if core.ex3.busy else IDLE
        s["ex4"] = ACTIVE if core.ex4.busy else IDLE
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
        rows = sa.algo_counts.get("act_rows", 0)
        s["array_in"] = ACTIVE if rows > prev["act_rows"] else IDLE
        prev["act_rows"] = rows
        if hasattr(sa, "_is_drained"):                  # MEISSA
            out = not sa._is_drained()
        else:
            out = any(row is not None for row in sa._output_pipe)
            s["psum_in"] = (IDLE if all(q.is_empty() for q in sa._psum_input_fifo_top)
                            else ACTIVE)
        s["array_out"] = ACTIVE if out else IDLE

        # Memory.
        for i, pad in enumerate(p.spad.tiles):
            s["spad%d" % i] = IDLE if pad.idle() else ACTIVE
            s["spad%d_w" % i] = ACTIVE if pad.wq else IDLE
            s["spad%d_r" % i] = ACTIVE if (pad.rq or pad.out) else IDLE
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
