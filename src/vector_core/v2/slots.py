"""Slot ops and rounds (plan section 3).

A SlotOp is what one slot V_k fires once: the destination it goes to
through the slot's crossbar, the registers it reads and writes, and what
the destination needs. The vector core never sees an instruction word;
whoever drives it (the scheduler's packet adapter today, a loop controller
or a kernel FSM later) builds SlotOps.

A round is up to S SlotOps, at most one per slot. Each fires once, when its
conditions hold; the round ends when all have fired, and the next round may
start the following cycle (the lockstep barrier, Q8). Single mode is one
round per packet.

Destinations (Q4): "fu:<group>", "vlsu0".."vlsu3", "gsau", "transpose", "mts".
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: What each kind of SlotOp does, and which destinations accept it.
KINDS = {
    "elem": "an element op on an FU group: vd = op(src0, src1 or scalar), masked",
    "compare": "a compare on the ALU group: mask vmd = op(src0, src1 or scalar)",
    "reduce": "a reduction through the ALU group and the RTree: vd = reduce(src0)",
    "load": "vreg.ld: vd = scratchpad row",
    "store": "vreg.st: scratchpad row = src0",
    "gemm": "gemm.vv: src0 to the systolic array, the result row to vd",
    "weight": "lw.vi: src0 to the systolic array as weights",
    "tpush": "tpus.vi: src0 into the transpose unit as a row",
    "tpop": "tpop.vi: the next transposed column to vd",
    "vts": "vmov.vts: element imm of src0 (fp32 bits) to scalar rd",
    "mts": "mv.mts: mask mask_reg to scalar rd",
}


@dataclass
class SlotOp:
    slot: int
    dest: str
    kind: str
    op: str = ""                         # element op, compare, or reduce op (sum/min/max)
    src0: Optional[int] = None           # vector registers read
    src1: Optional[int] = None
    scalar: Optional[float] = None       # .vs forms: the broadcast scalar
    mask_reg: Optional[int] = None       # the mask read (0 = m0, all ones); None = no mask read
    vd: Optional[int] = None
    vmd: Optional[int] = None
    rd: Optional[int] = None
    imm: int = 0
    vlsu: Dict[str, Any] = field(default_factory=dict)   # addr, vl, vst_rd
    name: str = ""                       # mnemonic, for stats and messages
    seq: int = 0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError("unknown slot op kind %r" % self.kind)

    def vector_reads(self) -> List[Tuple[str, int]]:
        """(operand, register) for each vector register read, in port order."""
        out = []
        if self.src0 is not None:
            out.append(("src0", self.src0))
        if self.src1 is not None and self.src1 != self.src0:
            out.append(("src1", self.src1))
        return out

    def vector_dst(self) -> Optional[int]:
        return self.vd if self.kind in ("elem", "reduce", "load", "gemm", "tpop") else None


@dataclass
class _OpState:
    """A SlotOp's progress through its round."""
    op: SlotOp
    reads_left: List[Tuple[str, int, int]]      # (operand, register, bank)
    mask_left: bool
    values: Dict[str, Any] = field(default_factory=dict)
    collected_at: Optional[int] = None
    send_ready: Optional[int] = None
    cross_pair: bool = False
    sent: bool = False
    reason: str = ""
