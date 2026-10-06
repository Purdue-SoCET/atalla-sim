"""What each scalar instruction computes: values, branch outcomes, addresses.

The execute stage needs real values -- a branch's direction, a load's
address -- so the model computes them itself; the functional sim stays a
test-time reference (scheduler/golden.py), and tests check these functions
against it.

Where the RTL, the functional sim and the ISA documents disagree, this
follows the ISA document (ASSEMBLY_SYNTAX.md) and the assembler (build.py),
which is what programs are written against, and otherwise the functional
sim. Each disagreement is listed in docs/scheduler-rtl-bugs.md:

  * Loads and stores address rs1 + imm. (The RTL drops rs1: execute_stage.sv
    drives ld_st_unit's address from the immediate alone.)
  * blt/bge/bgt/ble compare signed, as the ISA document says. (The
    functional sim keeps registers unsigned and compares unsigned.)
  * A branch's increment, [13:7], is unsigned 0..127, as the assembler
    encodes it. (scalar_control_unit.sv sign-extends it.)
  * jal jumps to pc + imm and jalr to rs1 + imm, byte offsets, as the
    assembler encodes them. (control.sv shifts both immediates left by 2.)
  * Scalar BF16 values live as fp32 bit patterns (BF16 in the upper half),
    and lhw.s / shw.s move the upper half of a register to and from a whole
    word, as in the functional sim. (The RTL keeps BF16 in the lower half
    and does halfword loads and stores.)
  * mod.s takes the sign of the divisor, as the functional sim's numpy `%`
    does. (The RTL's divider gives the sign of the dividend.)

Division by zero doesn't stop the machine, so the model doesn't raise (the
functional sim does): rcp.bf of +-0 is +-infinity, as IEEE gives, and
div.s / mod.s by zero follow RISC-V (quotient -1, remainder the dividend).
The RTL's divider result for a zero divisor hasn't been checked.
"""

import math
import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np

from scheduler.control import ScalarOp
from scheduler.isa import PACKET_BYTE_W, get_bits, sign_extend

MASK32 = 0xFFFFFFFF


def u32(x: int) -> int:
    return int(x) & MASK32


def s32(x: int) -> int:
    x = int(x) & MASK32
    return x - (1 << 32) if x & 0x80000000 else x


def fp32_bits(f: float) -> int:
    """functional_sim.fp32_to_hex."""
    return struct.unpack("<I", struct.pack("<f", f))[0]


def bits_fp32(x: int) -> float:
    """functional_sim.hex_to_fp32."""
    return struct.unpack("<f", struct.pack("<I", int(x) & MASK32))[0]


def bf16_operand(x: int) -> float:
    """functional_sim.scalar_reg_as_fp32_for_bf16_r_op: a register holding
    only 16 bits is a raw BF16 (the compiler's constant loads); anything
    wider is an fp32 bit pattern."""
    u = u32(x)
    return bits_fp32(u << 16) if u <= 0xFFFF else bits_fp32(u)


def mnemonic(op: ScalarOp) -> str:
    return op.mnemonic


# -- ALU and the multi-cycle units ------------------------------------------------

def _shift_amount(op: ScalarOp, b: int) -> int:
    return b & 31 if op.imm_src else u32(b)


def scalar_value(op: ScalarOp, rs1: int, rs2: int) -> int:
    """The value an ALU, multiply, divide, BF16 or convert op writes to rd.
    `rs1` and `rs2` are the register values decode 2 read."""
    m = op.mnemonic
    b = op.imm if op.imm_src else rs2
    base = m[:-3] if m.endswith("i.s") and m not in ("lui.s",) else m[:-2]

    if m == "lui.s":
        return u32(op.imm << 7)
    if base == "add":
        return u32(s32(rs1) + s32(b))
    if base == "sub":
        return u32(s32(rs1) - s32(b))
    if base == "mul":
        return u32(s32(rs1) * s32(b))
    if base == "div":
        if s32(b) == 0:
            return MASK32
        return u32(int(math.trunc(s32(rs1) / s32(b))))
    if base == "mod":
        if s32(b) == 0:
            return u32(rs1)
        return u32(s32(rs1) % s32(b))
    if base == "or":
        return u32(rs1 | u32(b))
    if base == "and":
        return u32(rs1 & u32(b))
    if base == "xor":
        return u32(rs1 ^ u32(b))
    if base == "sll":
        s = _shift_amount(op, b)
        return u32(u32(rs1) << s) if s < 32 else 0
    if base == "srl":
        s = _shift_amount(op, b)
        return u32(rs1) >> s if s < 32 else 0
    if base == "sra":
        s = _shift_amount(op, b)
        return u32(s32(rs1) >> min(s, 31))
    if base == "slt":
        return int(s32(rs1) < s32(b))
    if base == "sltu":
        return int(u32(rs1) < u32(b))

    if m in ("add.bf", "sub.bf", "mul.bf"):
        a, c = bf16_operand(rs1), bf16_operand(rs2)
        r = a + c if m == "add.bf" else a - c if m == "sub.bf" else a * c
        return fp32_bits(r)
    if m == "slt.bf":
        return int(bits_fp32(rs1) < bits_fp32(rs2))
    if m == "rcp.bf":
        a = bits_fp32(rs1)
        if a == 0.0:
            return fp32_bits(math.copysign(math.inf, a))
        return fp32_bits(1.0 / a)
    if m == "sqrt.bf":
        return fp32_bits(float(np.sqrt(bits_fp32(rs1))))
    if m == "stbf.s":
        return fp32_bits(float(np.float32(u32(rs1))))
    if m == "bfts.s":
        with np.errstate(invalid="ignore", over="ignore"):
            return u32(int(np.float32(bits_fp32(rs1)).astype(np.int32)))
    if m == "mv.stm":
        return u32(rs1)
    raise ValueError("no scalar value for %s" % m)


# -- control ------------------------------------------------------------------

BRANCH_TAKEN = {
    "beq.s": lambda a, b: a == b,
    "bne.s": lambda a, b: a != b,
    "blt.s": lambda a, b: s32(a) < s32(b),
    "bge.s": lambda a, b: s32(a) >= s32(b),
    "bgt.s": lambda a, b: s32(a) > s32(b),
    "ble.s": lambda a, b: s32(a) <= s32(b),
}


@dataclass(frozen=True)
class ControlOutcome:
    taken: bool               # jumps are always taken
    target: int              # where the next packet is, taken or not
    rd: Optional[int]        # register written, or None
    value: int = 0


def branch_offset(word: int) -> int:
    """BR-type: a signed 10-bit word offset, {[14], [39:31]}, in bytes."""
    return sign_extend(((get_bits(word, 14, 14) << 9) | get_bits(word, 39, 31)) << 2, 12)


def branch_increment(word: int) -> int:
    """BR-type [13:7], unsigned, as build.py's parse_incr_imm7 encodes it."""
    return get_bits(word, 13, 7)


def control_outcome(op: ScalarOp, pc: int, rs1: int, rs2: int) -> ControlOutcome:
    m = op.mnemonic
    fall = u32(pc + PACKET_BYTE_W)
    if m in BRANCH_TAKEN:
        taken = BRANCH_TAKEN[m](u32(rs1), u32(rs2))
        target = u32(pc + branch_offset(op.word)) if taken else fall
        return ControlOutcome(taken, target, op.rs1,
                              u32(rs1 + branch_increment(op.word)))
    if m == "jal":
        return ControlOutcome(True, u32(pc + sign_extend(get_bits(op.word, 39, 15), 25)),
                              op.rd, fall)
    if m == "jalr":
        return ControlOutcome(True, u32(rs1 + op.imm), op.rd, fall)
    raise ValueError("%s is not a control op" % m)


def is_control(op: ScalarOp) -> bool:
    return op.mnemonic in BRANCH_TAKEN or op.mnemonic in ("jal", "jalr")


# -- memory -----------------------------------------------------------------------

def mem_address(op: ScalarOp, rs1: int) -> int:
    return u32(rs1 + op.imm)


def store_word(op: ScalarOp, data: int) -> int:
    """The word a store writes: sw.s the register, shw.s its upper half."""
    return u32(data) >> 16 if op.mnemonic == "shw.s" else u32(data)


def load_value(op: ScalarOp, word: int) -> int:
    """What a load writes to rd: lw.s the word, lhw.s the word shifted up."""
    return u32(word << 16) if op.mnemonic == "lhw.s" else u32(word)
