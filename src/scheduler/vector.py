"""Vector and DMA dispatch: from the D2/EX latch to the vector units and back.

Transcribes scheduler_core.sv's ASSIGN_VECTOR_SPAD_INSTRS_TO_FUs,
move_to_scalar.sv and writeback/v_wb_arbiter.sv (atalla
scheduler_integration_SP26_joshklug), driving the vector core's unit models
(vector_core/) and the scratchpad backends (memory/backend.py).

Decode 2 has already read every operand -- vector registers, masks, scalars
-- so each unit is handed data. In the packet's EX cycle:

    lane ops (add, sub, mul, compares, exp,   up to 2 per packet, to the lane
    reductions)                                 datapath
    gemm.vv, lw.vi                            the GSAU
    vreg.ld, vreg.st                          the VLSU of scratchpad `sid`
    vmov.vts, mv.mts                          move-to-scalar: combinational,
                                              written through scalar writeback
    scpad.ld, scpad.st                        the backend of scratchpad
                                              rs3[31:30]

Readiness back to decode 2 is per unit: the lanes' ALU, MUL and EXP, the
reduction, the GSAU, move-to-scalar, each VLSU, and each scratchpad's busy
flag (an SDMA in flight).

Writeback (v_wb_arbiter): one vector register write per bank (vd[1:0]) per
cycle, by fixed priority VLSU 0-3, GSAU, reduction, lanes; a loser holds its
result. Compares write the mask file. A write offered in cycle v reaches the
register file in v + 1, through the EX/WB latch. An SDMA's completion clears
the busy bit on its rs1 the same way.

Where the RTL is wrong this does what was meant (docs/scheduler-rtl-bugs.md):
all four VLSUs write back (bug 4), the write ports are chosen by the
register file's bank vd[1:0] (bug 5), and lw.vi writes no register (bug 3).

Values follow the functional sim's lane rules (lane_values below): operands
are rounded to BF16, the op is done in fp32 and the result rounded to BF16
(nearest even); compares use the values as they are; `.vs` operands are the
scalar register's fp32 bit pattern; masked-off lanes keep the destination's
old value; reductions sum or compare the active lanes in lane order and place
the result by imm[6:5] (broadcast, one element, or one element over vs1); and
vmov.vts moves an element's fp32 bit pattern. One difference: the functional
sim writes a reduction's fp32 sum into the vector register unrounded, but the
register file holds BF16, so here it is rounded.

The lane datapath model times the lane ops; their values are computed here,
because the vector core's BF16 cast is FP16 while numpy has no bfloat16.
Vector registers hold BF16 values as floats; the scratchpad holds their
16-bit patterns.
"""

import math
import struct
from collections import deque

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from scheduler.control import (
    V_ALU_ADD, V_ALU_MEQ, V_ALU_MGT, V_ALU_MLT, V_ALU_MNEQ, V_ALU_SUB, V_EXP, V_GSAU,
    V_MUL, V_MVMT, V_REDU, V_VLSU, VectorOp, SdmaOp)
from scheduler.execute import Result
from scheduler.semantics import bits_fp32, fp32_bits

MASK32 = 0xFFFFFFFF


# -- BF16 ------------------------------------------------------------------------------

def bf16_round(x: float) -> float:
    """Round a value to the nearest BF16 (functional sim's bf16_round)."""
    u = fp32_bits(float(x)) if math.isfinite(x) else fp32_bits(x)
    lsb = (u >> 16) & 1
    u = (u + 0x7FFF + lsb) & 0xFFFF0000
    return bits_fp32(u)


def bf16_bits(x: float) -> int:
    return fp32_bits(bf16_round(x)) >> 16


def bits_bf16(b: int) -> float:
    return bits_fp32((int(b) & 0xFFFF) << 16)


def lane_mask(m: int, n: int) -> List[bool]:
    return [bool((m >> i) & 1) for i in range(n)]


def merge(old: Sequence[float], new: Sequence[float], mask: int) -> List[float]:
    return [n if (mask >> i) & 1 else o for i, (o, n) in enumerate(zip(old, new))]


# -- ops ---------------------------------------------------------------------------------

#: vector_fu_enable_t -> the lane datapath's op.
LANE_OP = {V_ALU_ADD: "add", V_ALU_SUB: "sub", V_MUL: "mul", V_EXP: "exp",
           V_ALU_MGT: "gt", V_ALU_MLT: "lt", V_ALU_MEQ: "eq", V_ALU_MNEQ: "ne"}
COMPARES = (V_ALU_MGT, V_ALU_MLT, V_ALU_MEQ, V_ALU_MNEQ)
REDUCE_OP = {"rsum.vi": "sum", "rmin.vi": "min", "rmax.vi": "max"}

#: The decode-2 readiness name of each lane op's unit.
UNIT_OF = {V_ALU_ADD: "alu", V_ALU_SUB: "alu", V_MUL: "mul", V_EXP: "exp",
           V_ALU_MGT: "alu", V_ALU_MLT: "alu", V_ALU_MEQ: "alu", V_ALU_MNEQ: "alu",
           V_REDU: "reduction"}
#: ...and the lane model's functional unit behind it.
LANE_FU = {"alu": "alu", "mul": "alu", "exp": "exp", "reduction": "alu"}


def _q(a) -> np.ndarray:
    """functional sim bf16_round, on float32."""
    u = np.asarray(a, dtype=np.float32).view(np.uint32)
    u = (u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return u.view(np.float32)


_ARITH = {V_ALU_ADD: np.add, V_ALU_SUB: np.subtract, V_MUL: np.multiply}
_CMP = {V_ALU_MGT: np.greater, V_ALU_MLT: np.less, V_ALU_MEQ: np.equal,
        V_ALU_MNEQ: np.not_equal}


def lane_values(o: VectorOp, vs1: Sequence[float], vs2, scalar: Optional[float],
                mask: int):
    """What a lane op computes before masking: a vector, a compare's bit
    vector as an int, or a reduction's value (functional sim vector_lanes)."""
    a = np.asarray(vs1, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        if o.fu in _CMP:
            b = np.float32(scalar) if scalar is not None else np.asarray(vs2, dtype=np.float32)
            bits = _CMP[o.fu](a, b)
            return sum(1 << i for i, t in enumerate(bits) if t)
        if o.fu in _ARITH:
            b = _q(np.full_like(a, scalar) if scalar is not None else np.asarray(vs2, np.float32))
            return [float(x) for x in _q(_ARITH[o.fu](_q(a), b))]
        if o.fu == V_EXP:
            return [float(x) for x in _q(np.exp(_q(a)))]
        if o.fu == V_REDU:
            q = _q(a)
            act = [q[i] for i in range(len(q)) if (mask >> i) & 1]
            if o.mnemonic == "rsum.vi":
                acc = np.float32(0.0)
                for x in act:
                    acc = np.float32(acc + x)
                return float(acc)
            if o.mnemonic == "rmin.vi":
                return float(min(act)) if act else math.inf
            return float(max(act)) if act else -math.inf
    raise ValueError("not a lane op: %s" % o.mnemonic)


def reduce_output(imm: int, vs: Sequence[float], r: float) -> List[float]:
    """functional_sim.apply_imm_vector_op."""
    idx = imm & 0x1F
    if (imm >> 6) & 1:
        return [r] * len(vs)
    if (imm >> 5) & 1:
        return [r if i == idx else 0.0 for i in range(len(vs))]
    return [r if i == idx else v for i, v in enumerate(vs)]


@dataclass
class _LaneOp:
    """An op in the lane datapath and the write it will make."""
    op: VectorOp
    write: "VectorWrite"


@dataclass
class VectorWrite:
    """A result offered to vector (or mask) writeback."""
    source: str
    reg: int
    value: Any                         # a vector, or a 32-bit mask
    is_mask: bool = False


# -- move-to-scalar ---------------------------------------------------------------------

class MoveToScalar:
    """move_to_scalar.sv: combinational, holds its result while writeback
    is not ready."""

    def __init__(self):
        self.held: Optional[Result] = None

    def offer(self, op: Optional[VectorOp], operands: Dict) -> Optional[Result]:
        self._in = None
        if self.held is not None:
            return self.held
        if op is None:
            return None
        if op.mnemonic == "vmov.vts":
            value = fp32_bits(operands["vs1"][op.imm & 0x1F])
        else:                                       # mv.mts
            value = int(operands["vms"]) & MASK32
        self._in = Result(op.rd, value, source="mts")
        return self._in

    def ready(self, granted: bool) -> bool:
        return self.held is None and not (self._in is not None and not granted)

    def advance(self, granted: bool) -> None:
        if self.held is not None:
            if granted:
                self.held = None
        elif self._in is not None and not granted:
            self.held = self._in


# -- the vector side ---------------------------------------------------------------------

class VectorSide:
    """Everything between the D2/EX latch's vector and SDMA slots and the
    EX/WB latch. One per SchedulerCore.

    `vc` is a VectorCore, whose units this drives; `backends` the scratchpad
    backends, one per scratchpad id; `read_vreg` / `read_mask` read decode
    2's register files (to merge masked writes with what they overwrite).
    """

    VLSU_PRIORITY = ("vlsu0", "vlsu1", "vlsu2", "vlsu3", "gsau", "reduction", "lanes")

    def __init__(self, vc, backends: Sequence = (), *, read_vreg: Callable[[int], List[float]],
                 read_mask: Callable[[int], int], spad_row_bytes: int = 64,
                 dtype: Any = "bf16"):
        self.vc = vc
        self.backends = list(backends)
        self.read_vreg, self.read_mask = read_vreg, read_mask
        self.row_bytes = int(spad_row_bytes)
        self.dtype = dtype
        self.n = vc.vector_len
        self.mts = MoveToScalar()
        self._lane_ops: Dict[int, _LaneOp] = {}
        self._lane_results: Deque[VectorWrite] = deque()
        self._vlsu_holds: Dict[int, Deque] = {s: deque() for s in range(len(vc.vls_units))}
        self.sdma_busy = [False] * max(4, len(self.backends))
        self._sdma_done: List[int] = []           # rs1s whose SDMA finished
        self.stats = dict(lane_ops=0, gsau_ops=0, vlsu_loads=0, vlsu_stores=0,
                          sdma_loads=0, sdma_stores=0, vector_writes=0, mask_writes=0,
                          wb_conflict_cycles=0)

    # -- readiness, from registered state ------------------------------------------------
    def unit_ready(self) -> Tuple[Dict[str, bool], List[bool], List[bool]]:
        dp = self.vc.datapath
        idle_q = dp.pending_issue.is_empty()
        vec = {u: idle_q and dp._can_issue_to_all_lanes(fu) for u, fu in LANE_FU.items()}
        g = self.vc.gsau
        vec["gsau"] = not g.to_systolic.is_full() and not g.rd_queue.is_full()
        vec["movement"] = self.mts.held is None
        vlsu = [v.can_accept_issue() for v in self.vc.vls_units]
        return vec, vlsu, list(self.sdma_busy[:4])

    # -- writeback offers ------------------------------------------------------------------
    def offers(self) -> List[VectorWrite]:
        out = []
        for s, vls in enumerate(self.vc.vls_units):
            wb = vls.wb_q.peek()
            if wb is not None:
                data = [bits_bf16(b) for b in wb["data"]][: self.n]
                data += [0.0] * (self.n - len(data))
                out.append(VectorWrite("vlsu%d" % s, int(wb["vd"]), data))
        wb = self.vc.gsau.writebacks.peek()
        if wb is not None:
            out.append(VectorWrite("gsau", int(wb["dst"]),
                                   [bf16_round(x) for x in wb["data"]]))
        if self._lane_results:
            out.append(self._lane_results[0])
        return out

    def arbitrate(self, offers: List[VectorWrite]):
        """v_wb_arbiter: one vector write per bank by priority, mask writes
        on their own ports. Returns (granted sources, vector writes, mask writes)."""
        order = {s: i for i, s in enumerate(self.VLSU_PRIORITY)}
        used, granted, vw, mw = set(), set(), [], []
        for o in sorted(offers, key=lambda o: order.get(o.source, 99)):
            if o.is_mask:
                granted.add(o.source)
                mw.append(o)
                continue
            bank = o.reg % 4
            if bank in used:
                self.stats["wb_conflict_cycles"] += 1
                continue
            used.add(bank)
            granted.add(o.source)
            vw.append(o)
        return granted, vw, mw

    def retire(self, granted) -> None:
        """Pop what writeback took this cycle."""
        for s, vls in enumerate(self.vc.vls_units):
            if "vlsu%d" % s in granted:
                vls.pop_writeback()
        if "gsau" in granted:
            self.vc.gsau.pop_writeback()
        if self._lane_results and self._lane_results[0].source in granted:
            self._lane_results.popleft()

    # -- dispatch from the D2/EX latch ------------------------------------------------------
    def dispatch(self, issued) -> None:
        if issued is None:
            return
        pkt = issued.packet
        for o in pkt.vector:
            if not o.valid:
                continue
            ops = issued.operands.get(("vector", o.slot), {})
            if o.fu == V_MVMT:
                continue                         # handled by offer_mts()
            if o.fu == V_VLSU:
                self._dispatch_vlsu(o, ops)
            elif o.fu == V_GSAU:
                self._dispatch_gsau(o, ops)
            else:
                self._dispatch_lanes(o, ops)
        for o in pkt.sdma:
            if o.valid:
                self._dispatch_sdma(o, issued.operands.get(("sdma", o.slot), {}))

    def mts_op(self, issued) -> Tuple[Optional[VectorOp], Dict]:
        if issued is None:
            return None, {}
        for o in issued.packet.vector:
            if o.valid and o.fu == V_MVMT:
                return o, issued.operands.get(("vector", o.slot), {})
        return None, {}

    def _dispatch_lanes(self, o: VectorOp, ops: Dict) -> None:
        """The lane datapath times the op; its value is computed here. The
        destination's old value can be read now: nothing else may write it
        while this op is in flight (the scoreboard's WAW check)."""
        vs1 = list(ops["vs1"])
        mask = int(ops.get("vms", MASK32))
        scalar = vs2 = None
        if o.op2_src == 0:
            vs2 = list(ops["vs2"])
        elif o.op2_src == 2:                       # .vs: the scalar's fp32 bits
            scalar = bits_fp32(ops["rs1"])
        value = lane_values(o, vs1, vs2, scalar, mask)
        if o.fu == V_REDU:
            write = VectorWrite("reduction", o.vd,
                                reduce_output(o.imm, vs1, bf16_round(value)))
        elif o.fu in COMPARES:
            old = self.read_mask(o.vmd)
            write = VectorWrite("lanes", o.vmd, (value & mask) | (old & ~mask & MASK32),
                                is_mask=True)
        else:
            write = VectorWrite("lanes", o.vd, merge(self.read_vreg(o.vd), value, mask))
        inst = self.vc.datapath.enqueue(
            src0=vs1, src1=None, mask=None,
            op="exp" if o.fu == V_EXP else "add", dst=o.vd,
            reduce=o.fu == V_REDU, reduce_op="sum", reduce_out_mode="broadcast",
            dtype=self.dtype)
        self._lane_ops[inst] = _LaneOp(o, write)
        self.stats["lane_ops"] += 1

    def _collect_lanes(self) -> None:
        dp = self.vc.datapath
        if dp.result_valid:
            self._lane_results.append(self._lane_ops.pop(dp.last_result["inst_id"]).write)

    def _dispatch_gsau(self, o: VectorOp, ops: Dict) -> None:
        weight = o.mnemonic == "lw.vi"
        cmd = {"vdata": list(ops["vs1"]), "is_weight": weight,
               "expect_output": not weight, "dtype": self.dtype,
               "meta": {"dtype": self.dtype, "kind": o.mnemonic}}
        if not weight:
            cmd["dst"] = o.vd
        assert self.vc.gsau.issue(cmd), "GSAU issued while not ready"
        self.stats["gsau_ops"] += 1

    def _row(self, spad_addr: int, row: int = 0) -> int:
        return int(spad_addr) // self.row_bytes + int(row)

    def _dispatch_vlsu(self, o: VectorOp, ops: Dict) -> None:
        vls = self.vc.vls_units[o.sid]
        addr = self._row(ops["rs1"], ops["rs2"])
        if o.vector_reg_write:
            op = {"kind": "load", "scratchpad": 0, "vd": o.vd, "addr": addr,
                  "vl": o.num_cols + 1, "dtype": self.dtype}
            self.stats["vlsu_loads"] += 1
        else:
            data = [bf16_bits(x) for x in ops["vs1"]]
            op = {"kind": "store", "scratchpad": 0, "data": data, "addr": addr,
                  "vl": o.num_cols + 1}
            self.stats["vlsu_stores"] += 1
        assert vls.enqueue_issue(op), "VLSU %d issued while not ready" % o.sid

    def _dispatch_sdma(self, o: SdmaOp, ops: Dict) -> None:
        meta = int(ops["rs3"]) & MASK32
        sid = (meta >> 30) & 3
        rows, cols = ((meta >> 25) & 0x1F) + 1, ((meta >> 20) & 0x1F) + 1
        full = meta & 0xFFFFF
        stride = (full + 1 if full else cols) * 2
        be = self.backends[sid]
        rs1 = o.rs1_rd

        def done(_tx, _rs1=rs1, _sid=sid):
            self._sdma_done.append(_rs1)
            self.sdma_busy[_sid] = False

        start = be.driver_to_backend_start_store if o.store else be.driver_to_backend_start_load
        tx = start(self._row(ops["rs1"]), int(ops["rs2"]) & MASK32, rows, cols,
                   callback=done, dram_stride=stride)
        assert tx >= 0, "scratchpad %d backend refused an SDMA" % sid
        self.sdma_busy[sid] = True
        self.stats["sdma_stores" if o.store else "sdma_loads"] += 1

    # -- the edge ------------------------------------------------------------------------
    def tick(self, cycle: int) -> None:
        self.vc.tick_units(float(cycle))
        self._collect_lanes()

    def take_sdma_done(self) -> List[int]:
        done, self._sdma_done = self._sdma_done, []
        return done

    @property
    def idle(self) -> bool:
        vc = self.vc
        return (not self._lane_ops and not self._lane_results
                and vc.datapath.pending_issue.is_empty()
                and not vc.gsau.has_pending() and vc.gsau.rd_queue.is_empty()
                and vc.gsau.writebacks.is_empty()
                and all(v.issue_q.is_empty() and v.req_q.is_empty() and v.rsp_q.is_empty()
                        and v.wb_q.is_empty() and v.outstanding_loads() == 0
                        for v in vc.vls_units)
                and not any(self.sdma_busy) and self.mts.held is None)
