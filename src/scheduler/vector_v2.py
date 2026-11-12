"""The vector side for the M x N vector core (vector_core/v2): the packet
adapter of plan milestone M7.

Each packet decode 2 issues becomes one round: its vector word in slot k
goes to slot V_k (single mode, trip 1). Decode 2 no longer reads vector
registers or masks (Decode2(vector_operands_at_issue=False)); it passes
register numbers and the scalar values it read, and the slots read the
banks themselves. Vector-register hazards are the vector core's; decode 2
keeps the scalar and mask busy bits.

Writeback of vector registers happens inside the vector core. Compare
results (masks) leave it through `on_mask_write` and go to decode 2's mask
file through the EX/WB latch (writeback_at), which clears their busy bits.
A cycle's scalar-side mask writes (mv.stm) take their mask bank first; the
compare waits (worklog: the reverse of Q21).

Move-to-scalar: the vector core's one-entry hold register is offered to
scalar writeback (MoveToScalarV2), one cycle after the slot fires
(worklog: Q19 says the same cycle).

SDMA dispatch and the vreg.st reservation release are v1's (VectorSide).
"""

from collections import deque
from typing import Callable, Dict, List, Optional, Sequence

from scheduler.control import (
    V_ALU_ADD, V_ALU_MEQ, V_ALU_MGT, V_ALU_MLT, V_ALU_MNEQ, V_ALU_SUB, V_EXP, V_GSAU, V_MUL,
    V_MVMT, V_REDU, V_TRANS, V_VLSU, VectorOp)
from scheduler.execute import Result
from scheduler.semantics import bits_fp32
from scheduler.vector import REDUCE_OP, VectorSide
from vector_core.v2.slots import SlotOp

ELEM_OP = {V_ALU_ADD: "add", V_ALU_SUB: "sub", V_MUL: "mul", V_EXP: "exp"}
COMPARE_OP = {V_ALU_MGT: "gt", V_ALU_MLT: "lt", V_ALU_MEQ: "eq", V_ALU_MNEQ: "ne"}


class MoveToScalarV2:
    """The vector core's move-to-scalar hold register, as scalar writeback
    sees it. Readiness lives in the vector core (the "mts" destination)."""

    def __init__(self, vc, round_room: Callable[[], bool]):
        self.vc = vc
        self.round_room = round_room

    @property
    def held(self) -> Optional[Result]:
        h = self.vc.mts_hold
        return None if h is None else Result(h[0], h[1], source="mts")

    def offer(self, op, operands) -> Optional[Result]:
        return self.held

    def ready(self, granted: bool) -> bool:
        # SchedulerCore takes decode 2's "movement" readiness from here: a
        # move-to-scalar packet needs room for its round like any other.
        return self.round_room()

    def advance(self, granted: bool) -> None:
        if granted and self.vc.mts_hold is not None:
            self.vc.release_mts()


class VectorSideV2:
    """Everything between the D2/EX latch's vector and SDMA slots and the
    M x N vector core. One per SchedulerCore."""

    # v1's SDMA and vreg.st handling, unchanged
    _dispatch_sdma = VectorSide._dispatch_sdma
    _row = VectorSide._row
    take_sdma_done = VectorSide.take_sdma_done
    vst_committed = VectorSide.vst_committed
    take_vst_done = VectorSide.take_vst_done

    def __init__(self, vc, backends: Sequence = (), *, read_mask: Callable[[int], int],
                 writeback_at: Callable, spad_row_bytes: int = 64):
        self.vc = vc
        self.backends = list(backends)
        self.row_bytes = int(spad_row_bytes)
        self.n = vc.vector_len
        self.mts = MoveToScalarV2(vc, self._round_room)
        self.sdma_busy = [False] * max(4, len(self.backends))
        self._sdma_done: List[int] = []
        self._vst_done: List[int] = []
        self._lane_results: List = []                 # PerfMonitor reads it
        #: Mask banks the scalar side writes this cycle (set by the core).
        self.blocked_mask_banks = set()
        vc.read_mask = read_mask
        vc.on_mask_write = lambda cycle, m, value: writeback_at(cycle + 1, mask=[(m, value)])
        self.stats = dict(lane_ops=0, gsau_ops=0, vlsu_loads=0, vlsu_stores=0,
                          transpose_pushes=0, transpose_pops=0, sdma_loads=0, sdma_stores=0,
                          vector_writes=0, mask_writes=0, wb_conflict_cycles=0, rounds=0)

    # -- what PerfMonitor reads ----------------------------------------------------
    @property
    def _lane_ops(self) -> bool:
        return self.vc.fu_busy

    @property
    def _tp_reqs(self):
        return self.vc.tp_reqs

    @property
    def _tp_vds(self):
        return self.vc.tp_vds

    # -- readiness, from registered state --------------------------------------------
    def _round_room(self) -> bool:
        # The D2/EX latch may already hold a round that dispatch adds this
        # cycle, so a new packet needs room for two.
        return len(self.vc.rounds) + 2 <= self.vc.cfg.round_queue_depth

    def unit_ready(self):
        ok = self._round_room()
        vec = {u: ok for u in ("alu", "mul", "exp", "reduction", "gsau", "movement",
                               "transpose")}
        vlsu = [ok] * max(4, len(self.vc.vls_units))
        return vec, vlsu, list(self.sdma_busy[:4])

    # -- writeback: inside the vector core ---------------------------------------------
    def offers(self):
        return []

    def arbitrate(self, offers):
        return set(), [], []

    def retire(self, granted) -> None:
        pass

    def mts_op(self, issued):
        return None, {}

    # -- dispatch from the D2/EX latch ------------------------------------------------
    def dispatch(self, issued) -> None:
        if issued is None:
            return
        pkt = issued.packet
        ops = []
        for o in pkt.vector:
            if o.valid:
                ops.append(self.slot_op(o, issued.operands.get(("vector", o.slot), {})))
        if ops:
            self.vc.issue_round(ops)
            self.stats["rounds"] += 1
        for o in pkt.sdma:
            if o.valid:
                self._dispatch_sdma(o, issued.operands.get(("sdma", o.slot), {}))

    def slot_op(self, o: VectorOp, ops: Dict) -> SlotOp:
        """One decoded vector instruction as slot V_k's op (k = its packet slot)."""
        cfg = self.vc.cfg
        mask = o.vms if o.use_vms else None
        base = dict(slot=o.slot, name=o.mnemonic)
        scalar = bits_fp32(ops["rs1"]) if o.op2_src == 2 else None
        src1 = o.vs2 if (o.op2_src == 0 and o.use_vs2) else None
        if o.fu in ELEM_OP:
            op = ELEM_OP[o.fu]
            self.stats["lane_ops"] += 1
            return SlotOp(dest="fu:" + cfg.group_for_op(op), kind="elem", op=op, src0=o.vs1,
                          src1=src1, scalar=scalar, mask_reg=mask, vd=o.vd, **base)
        if o.fu in COMPARE_OP:
            op = COMPARE_OP[o.fu]
            self.stats["lane_ops"] += 1
            return SlotOp(dest="fu:" + cfg.group_for_op(op), kind="compare", op=op,
                          src0=o.vs1, src1=src1, scalar=scalar, mask_reg=mask, vmd=o.vmd, **base)
        if o.fu == V_REDU:
            self.stats["lane_ops"] += 1
            alu = next(g.name for g in cfg.fu_groups if g.rtree)
            return SlotOp(dest="fu:" + alu, kind="reduce", op=REDUCE_OP[o.mnemonic],
                          src0=o.vs1, mask_reg=mask, vd=o.vd, imm=o.imm, **base)
        if o.fu == V_VLSU:
            vlsu = {"addr": self._row(ops["rs1"], ops["rs2"]), "vl": o.num_cols + 1}
            if o.vector_reg_write:
                self.stats["vlsu_loads"] += 1
                return SlotOp(dest="vlsu%d" % o.sid, kind="load", vd=o.vd, vlsu=vlsu, **base)
            self.stats["vlsu_stores"] += 1
            vlsu["vst_rd"] = o.rs1
            return SlotOp(dest="vlsu%d" % o.sid, kind="store", src0=o.vs1, vlsu=vlsu, **base)
        if o.fu == V_GSAU:
            self.stats["gsau_ops"] += 1
            if o.mnemonic == "lw.vi":
                return SlotOp(dest="gsau", kind="weight", src0=o.vs1, **base)
            return SlotOp(dest="gsau", kind="gemm", src0=o.vs1, vd=o.vd, **base)
        if o.fu == V_TRANS:
            if o.mnemonic == "tpus.vi":
                self.stats["transpose_pushes"] += 1
                return SlotOp(dest="transpose", kind="tpush", src0=o.vs1, **base)
            self.stats["transpose_pops"] += 1
            return SlotOp(dest="transpose", kind="tpop", vd=o.vd, **base)
        if o.fu == V_MVMT:
            if o.mnemonic == "vmov.vts":
                return SlotOp(dest="mts", kind="vts", src0=o.vs1, rd=o.rd, imm=o.imm, **base)
            return SlotOp(dest="mts", kind="mts", mask_reg=o.vms, rd=o.rd, **base)
        raise ValueError("no slot op for %s" % o.mnemonic)

    # -- the edge ------------------------------------------------------------------------
    def tick(self, cycle: int) -> None:
        self.vc.blocked_mask_banks = frozenset(self.blocked_mask_banks)
        self.vc.step(cycle)
        self.vc.tick_units(float(cycle))

    @property
    def idle(self) -> bool:
        return self.vc.idle and not any(self.sdma_busy)
