"""Decode 2's control units: what each instruction reads, writes and needs.

Transcribes rtl/modules/scheduler/decode2/{scalar,vector,sdma}_control_unit.sv
(atalla scheduler_integration_SP26_joshklug). Each turns a raw 40-bit word
into the fields the rest of decode 2 works from -- the registers it reads,
the registers it writes, the functional unit it needs, its immediates --
taking every bit slice from the RTL, not from the ISA sheet.

li.s (47) is a pseudo-instruction -- lui.s then addi.s -- that the
assembler expands, so it never reaches the hardware, and scalar_control_unit
rightly has no case for it: it decodes as invalid here too. lui.s is real:
rd[31:7] = imm[24:0], the shift done by the ALU. (The functional sim's
assembler currently emits a raw li.s and runs it natively; a program built
that way would have it dropped by the hardware.)

Where the RTL is wrong, this module does what was meant and says so; it
does not reproduce the bug:

  * lw.vi sets vector_reg_write with vd = [14:7], but a weight load writes
    no vector register (the GSAU only raises weight_en). The scoreboard
    would mark vd busy and nothing would ever clear it. Here it writes none.
  * A branch's increment, [13:7], is unsigned 0..127 -- the assembler
    rejects anything else -- but the RTL sign-extends it. Here it is unsigned.
  * sqrt.bf's unit, sqrt_valid, is routed to EX2 by the execute crossbar
    but missing from decode_2's EX2 structural check, so a sqrt could issue
    into a busy EX2. Here it needs EX2 like the other EX2 operations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from scheduler.isa import OP_W, OPCODES, get_bits, sign_extend

_BY_NAME = {m: op for op, (m, _) in OPCODES.items()}


def _op(name: str) -> int:
    return _BY_NAME[name]


# -- scalar_fu_enable_t ----------------------------------------------------------
ALU, CONTROL, BF_ADD, BF_SUB, BF_MULT, BF_SLT, BF_DIV, S_DIV, S_MOD, \
    BF_TO_INT, INT_TO_BF, LD, ST, S_MULT, HALT, SQRT = range(16)

#: The execute crossbar's routing (xbar_4x5_exec_comb.sv decode_dest), with
#: sqrt in EX2 where the crossbar already sends it.
EX_OF_SCALAR_FU = {
    ALU: 1, CONTROL: 1,
    BF_DIV: 2, S_DIV: 2, S_MOD: 2, BF_TO_INT: 2, INT_TO_BF: 2, SQRT: 2,
    BF_ADD: 3, BF_SUB: 3, BF_MULT: 3, BF_SLT: 3,
    S_MULT: 4,
    LD: 5, ST: 5,
}

# -- vector_fu_enable_t ----------------------------------------------------------
(V_ALU_ADD, V_ALU_SUB, V_ALU_AND, V_ALU_OR, V_ALU_XOR, V_ALU_NOT, V_ALU_MGT,
 V_ALU_MLT, V_ALU_MEQ, V_ALU_MNEQ, V_MUL, V_EXP, V_VLSU, V_GSAU, V_REDU,
 V_MVMT) = range(16)
VECTOR_ALU_FUS = frozenset(range(V_ALU_ADD, V_ALU_MNEQ + 1))


@dataclass
class ScalarOp:
    slot: int
    word: int
    op: int
    valid: bool = False
    fu: int = ALU
    rs1: int = 0
    rs2: int = 0
    rd: int = 0
    use_rs1: bool = False
    use_rs2: bool = False
    reg_write: bool = False
    mask_reg_write: bool = False
    imm: int = 0
    imm_src: bool = False
    halfword: bool = False
    incr7: int = 0
    halt: bool = False

    @property
    def mnemonic(self) -> str:
        return OPCODES.get(self.op, ("unknown", ""))[0]

    @property
    def ex(self) -> Optional[int]:
        """Execute unit 1..5, or None (halt, or not a valid instruction)."""
        if not self.valid:
            return None
        if self.mask_reg_write:
            return 2
        return EX_OF_SCALAR_FU.get(self.fu)

    def reads(self) -> List[int]:
        return ([self.rs1] if self.use_rs1 else []) + ([self.rs2] if self.use_rs2 else [])

    def scalar_write(self) -> Optional[int]:
        return self.rd if self.valid and self.reg_write else None

    def mask_write(self) -> Optional[int]:
        """mv.stm: the dependency checker marks rdIn[3:0] busy."""
        return self.rd & 0xF if self.valid and self.mask_reg_write else None


@dataclass
class VectorOp:
    slot: int
    word: int
    op: int
    valid: bool = False
    fu: int = 0
    op2_src: int = 0
    vms: int = 0
    vmd: int = 0
    vs1: int = 0
    vs2: int = 0
    vd: int = 0
    rs1: int = 0
    rs2: int = 0
    rd: int = 0
    imm: int = 0
    sid: int = 0
    num_cols: int = 0
    rm: bool = False
    use_vms: bool = False
    use_vs1: bool = False
    use_vs2: bool = False
    use_rs1: bool = False
    use_rs2: bool = False
    scalar_reg_write: bool = False
    vector_reg_write: bool = False
    mask_reg_write: bool = False

    @property
    def mnemonic(self) -> str:
        return OPCODES.get(self.op, ("unknown", ""))[0]

    def scalar_reads(self) -> List[int]:
        return ([self.rs1] if self.use_rs1 else []) + ([self.rs2] if self.use_rs2 else [])

    def vector_reads(self) -> List[int]:
        return ([self.vs1] if self.use_vs1 else []) + ([self.vs2] if self.use_vs2 else [])

    def mask_reads(self) -> List[int]:
        return [self.vms] if self.use_vms else []

    def scalar_write(self) -> Optional[int]:
        return self.rd if self.valid and self.scalar_reg_write else None

    def vector_write(self) -> Optional[int]:
        return self.vd if self.valid and self.vector_reg_write else None

    def mask_write(self) -> Optional[int]:
        return self.vmd if self.valid and self.mask_reg_write else None

    @property
    def is_lane_op(self) -> bool:
        """What scheduler_core routes to the lanes: everything but GSAU,
        VLSU and MVMT."""
        return self.valid and self.fu not in (V_GSAU, V_VLSU, V_MVMT)


@dataclass
class SdmaOp:
    slot: int
    word: int
    op: int
    valid: bool = False
    store: bool = False
    rs1_rd: int = 0
    rs2: int = 0
    rs3: int = 0
    use_rs1: bool = False
    use_rs2: bool = False
    use_rs3: bool = False

    @property
    def mnemonic(self) -> str:
        return OPCODES.get(self.op, ("unknown", ""))[0]

    def reads(self) -> List[int]:
        return ([self.rs1_rd] if self.use_rs1 else []) + \
               ([self.rs2] if self.use_rs2 else []) + \
               ([self.rs3] if self.use_rs3 else [])

    def scalar_write(self) -> Optional[int]:
        """An SDMA reserves rs1 (= rd) until the scratchpad reports it done."""
        return self.rs1_rd if self.valid and self.use_rs1 else None


# -- scalar_control_unit -----------------------------------------------------------
def _i12(w: int) -> int:
    return sign_extend(get_bits(w, 34, 23), 12)


_R_ALU = {"add.s", "sub.s", "or.s", "and.s", "xor.s", "sll.s", "srl.s", "sra.s",
          "slt.s", "sltu.s"}
_I_ALU = {"addi.s", "subi.s", "ori.s", "andi.s", "xori.s", "slli.s", "srli.s",
          "srai.s", "slti.s", "sltui.s"}
# R-type: fu, halfword, use_rs2
_R_OTHER = {
    "mul.s": (S_MULT, False, True), "div.s": (S_DIV, False, True),
    "mod.s": (S_MOD, False, True), "add.bf": (BF_ADD, True, True),
    "sub.bf": (BF_SUB, True, True), "mul.bf": (BF_MULT, True, True),
    "rcp.bf": (BF_DIV, True, False), "slt.bf": (BF_SLT, True, True),
    "sqrt.bf": (SQRT, True, False), "stbf.s": (INT_TO_BF, True, False),
    "bfts.s": (BF_TO_INT, False, False),
}
_I_OTHER = {"muli.s": S_MULT, "divi.s": S_DIV, "modi.s": S_MOD}
_BRANCHES = {"beq.s", "bne.s", "blt.s", "bge.s", "bgt.s", "ble.s"}


def decode_scalar(word: int, slot: int = 0) -> ScalarOp:
    op = get_bits(word, OP_W - 1, 0)
    d = ScalarOp(slot=slot, word=word, op=op, rs1=get_bits(word, 22, 15),
                 rs2=get_bits(word, 30, 23), rd=get_bits(word, 14, 7))
    m = OPCODES.get(op, ("", ""))[0]
    if m in _R_ALU:
        d.valid, d.fu, d.use_rs1, d.use_rs2, d.reg_write = True, ALU, True, True, True
    elif m in _R_OTHER:
        fu, half, rs2 = _R_OTHER[m]
        d.valid, d.fu, d.halfword = True, fu, half
        d.use_rs1, d.use_rs2, d.reg_write = True, rs2, True
    elif m == "mv.stm":
        d.valid, d.fu, d.use_rs1, d.mask_reg_write = True, BF_TO_INT, True, True
    elif m in _I_ALU or m in _I_OTHER:
        d.valid, d.fu = True, (ALU if m in _I_ALU else _I_OTHER[m])
        d.imm_src, d.imm = True, _i12(word)
        d.use_rs1, d.reg_write = True, True
    elif m in _BRANCHES:
        d.valid, d.fu, d.imm_src = True, CONTROL, True
        d.imm = sign_extend((get_bits(word, 14, 14) << 9) | get_bits(word, 39, 31), 10)
        d.incr7 = get_bits(word, 13, 7)        # unsigned, as build.py encodes it
        d.use_rs1, d.use_rs2, d.reg_write = True, True, True
        d.rd = d.rs1                           # rs1 += incr7
    elif m == "jal":
        d.valid, d.fu, d.imm_src, d.reg_write = True, CONTROL, True, True
        d.imm = sign_extend(get_bits(word, 39, 15), 25)
    elif m == "jalr":
        d.valid, d.fu, d.imm_src = True, CONTROL, True
        d.imm, d.use_rs1, d.reg_write = _i12(word), True, True
    elif m in ("lw.s", "lhw.s"):
        d.valid, d.fu, d.imm_src, d.halfword = True, LD, True, m == "lhw.s"
        d.imm, d.use_rs1, d.reg_write = _i12(word), True, True
    elif m in ("sw.s", "shw.s"):
        d.valid, d.fu, d.imm_src, d.halfword = True, ST, True, m == "shw.s"
        d.imm, d.use_rs1, d.use_rs2 = _i12(word), True, True
        d.rs2 = get_bits(word, 14, 7)          # the data register
    elif m == "lui.s":                         # rd[31:7] = imm[24:0]
        d.valid, d.fu, d.imm_src, d.reg_write = True, ALU, True, True
        d.imm = get_bits(word, 39, 15)
    elif m == "halt.s":
        d.valid, d.fu, d.halt = True, HALT, True
    # nop.s and anything else: not valid, does nothing
    return d


# -- vector_control_unit -----------------------------------------------------------
_VV = {"add.vv": V_ALU_ADD, "sub.vv": V_ALU_SUB, "mul.vv": V_MUL}
_MVV = {"mgt.mvv": V_ALU_MGT, "mlt.mvv": V_ALU_MLT, "meq.mvv": V_ALU_MEQ,
        "mneq.mvv": V_ALU_MNEQ}
_VS = {"add.vs": V_ALU_ADD, "sub.vs": V_ALU_SUB, "mul.vs": V_MUL}
_MVS = {"mgt.mvs": V_ALU_MGT, "mlt.mvs": V_ALU_MLT, "meq.mvs": V_ALU_MEQ,
        "mneq.mvs": V_ALU_MNEQ}


def decode_vector(word: int, slot: int = 0) -> VectorOp:
    op = get_bits(word, OP_W - 1, 0)
    d = VectorOp(slot=slot, word=word, op=op)
    m = OPCODES.get(op, ("", ""))[0]
    vms, vs1, vs2, vd = (get_bits(word, 34, 31), get_bits(word, 22, 15),
                         get_bits(word, 30, 23), get_bits(word, 14, 7))
    if m in _VV:
        d.valid, d.fu, d.vms, d.vs1, d.vs2, d.vd = True, _VV[m], vms, vs1, vs2, vd
        d.use_vms, d.use_vs1, d.use_vs2, d.vector_reg_write = True, True, True, True
    elif m == "gemm.vv":
        d.valid, d.fu, d.vms, d.vs1, d.vd = True, V_GSAU, vms, vs1, vd
        d.use_vs1, d.vector_reg_write = True, True
    elif m in _MVV:
        d.valid, d.fu, d.vms, d.vmd, d.vs1, d.vs2 = (True, _MVV[m], vms,
                                                     get_bits(word, 10, 7), vs1, vs2)
        d.use_vms, d.use_vs1, d.use_vs2, d.mask_reg_write = True, True, True, True
    elif m in ("expi.vi", "lw.vi", "rsum.vi", "rmin.vi", "rmax.vi"):
        d.valid, d.imm, d.op2_src = True, get_bits(word, 30, 23), 1
        d.vms, d.vs1, d.vd = vms, vs1, vd
        d.use_vms, d.use_vs1 = True, True
        d.fu = {"expi.vi": V_EXP, "lw.vi": V_GSAU}.get(m, V_REDU)
        d.rm = d.fu == V_REDU
        d.vector_reg_write = m != "lw.vi"      # a weight load writes no vreg
    elif m in _VS:
        d.valid, d.fu, d.op2_src = True, _VS[m], 2
        d.vms, d.vs1, d.vd, d.rs1 = vms, vs1, vd, get_bits(word, 30, 23)
        d.use_vms, d.use_vs1, d.use_rs1, d.vector_reg_write = True, True, True, True
    elif m in _MVS:
        d.valid, d.fu, d.op2_src = True, _MVS[m], 2
        d.vms, d.vmd, d.vs1, d.rs1 = vms, get_bits(word, 10, 7), vs1, get_bits(word, 30, 23)
        d.use_vms, d.use_vs1, d.use_rs1, d.mask_reg_write = True, True, True, True
    elif m == "vmov.vts":
        d.valid, d.fu, d.imm, d.op2_src = True, V_MVMT, get_bits(word, 30, 23), 1
        d.vs1, d.rd, d.use_vs1, d.scalar_reg_write = vs1, vd, True, True
    elif m == "mv.mts":
        d.valid, d.fu, d.vms, d.rd = True, V_MVMT, get_bits(word, 18, 15), vd
        d.use_vms, d.scalar_reg_write = True, True
    elif m in ("vreg.ld", "vreg.st"):
        d.valid, d.fu = True, V_VLSU
        d.rs1, d.rs2 = get_bits(word, 22, 15), get_bits(word, 30, 23)
        d.sid, d.num_cols = get_bits(word, 37, 36), get_bits(word, 35, 31)
        d.use_rs1, d.use_rs2 = True, True
        if m == "vreg.ld":
            d.vd, d.vector_reg_write = vd, True
        else:
            d.vs1, d.use_vs1 = vd, True        # the stored register sits in [14:7]
    return d


# -- sdma_control_unit ---------------------------------------------------------------
def decode_sdma(word: int, slot: int = 0) -> SdmaOp:
    op = get_bits(word, OP_W - 1, 0)
    m = OPCODES.get(op, ("", ""))[0]
    if m not in ("scpad.ld", "scpad.st"):
        return SdmaOp(slot=slot, word=word, op=op)
    return SdmaOp(slot=slot, word=word, op=op, valid=True, store=m == "scpad.st",
                  rs1_rd=get_bits(word, 14, 7), rs2=get_bits(word, 22, 15),
                  rs3=get_bits(word, 30, 23), use_rs1=True, use_rs2=True, use_rs3=True)
