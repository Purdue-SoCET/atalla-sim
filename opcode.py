from enum import Enum, auto
from helpers import tobits, frombits
import numpy as np

# Opcodes
# - instruction formats in the RISCV Spec Chapter 34 
class Opcode(Enum):
    NOP = auto()
    # MOV = auto() not in the RISCV Spec
    
    # R Type
    RTYPE = auto()

    # I Type
    ITYPE = auto()
    
    SLTUI = auto()
    
    LW = auto()
    
    JALR = auto()
    
    # S Type
    SW = auto()
    
    # B Type
    BTYPE = auto()
    
    # U Type
    LUI = auto()
    
    # J Type
    JAL = auto()
    
    # MUL Extension
    MUL = auto()

    # Matrix Extension
    STM = auto()
    LDM = auto()
    GEMM = auto()
    
    # Macros
    PUSH = auto()
    POP = auto()
    HALT = auto()
    
class AluOp(Enum):
    NOP = auto()
    ADD = auto()
    SUB = auto()
    XOR = auto()
    OR = auto()
    AND = auto()
    SLL = auto()
    SRL = auto()
    SLT = auto()
    SLTU = auto()
    SRA = auto()

alu_funct = {
    AluOp.NOP: lambda a, b: a,
    AluOp.ADD: lambda a, b: np.int32(np.int32(a) + np.int32(b)),
    AluOp.SUB: lambda a, b: np.int32(np.int32(a) - np.int32(b)),
    AluOp.XOR: lambda a, b: a^b,
    AluOp.OR: lambda a,b: a|b,
    AluOp.AND: lambda a,b: a&b,
    AluOp.SLL: lambda a,b: frombits(b*[0] + tobits([a], 32)[:32-b], False),
    AluOp.SRL: lambda a,b: frombits(tobits([a], 32)[b:] + b*[0], False),
    AluOp.SLT: lambda a,b: a < b,
    AluOp.SLTU: lambda a,b: np.int32(np.uint32(a) < np.uint32(b)),
    AluOp.SRA: lambda a,b: a >> b,
}

rfunct = {
    0b0000000000: AluOp.ADD,
    0b0000100000: AluOp.SUB,
    0b1000000000: AluOp.XOR,
    0b1100000000: AluOp.OR,
    0b1110000000: AluOp.AND,
    0b0010000000: AluOp.SLL,
    0b1010000000: AluOp.SRL,
    0b1010100000: AluOp.SRA,
    0b0100000000: AluOp.SLT,
    0b0110000000: AluOp.SLTU, 
}
ifunct = {
    0b000: AluOp.ADD,
    0b010: AluOp.ADD,
    0b100: AluOp.XOR,
    0b110: AluOp.OR,
    0b111: AluOp.AND,
    0b001: AluOp.SLL, # imm = 0x00
    0b101: AluOp.SRL, # imm = 0x20
    # 0b101: AluOp.SRA, imm = 0x20
    0b011: AluOp.SLTU,
    
}

class BranchOp(Enum):
    BEQ = auto()
    BNE = auto()
    BLT = auto()
    BGE = auto()
    
branch_funct = {
    BranchOp.BEQ: lambda a: a == 0,
    BranchOp.BNE: lambda a: a != 0,
    BranchOp.BLT: lambda a: a < 0,
    BranchOp.BGE: lambda a: a >= 0,
}
    
bfunct = {
    0b000: BranchOp.BEQ,
    0b001: BranchOp.BNE,
    0b100: BranchOp.BLT,
    0b101: BranchOp.BGE
}

opcodes = {
    0b0110011: Opcode.RTYPE,
    0b0010011: Opcode.ITYPE,
    0b0000011: Opcode.LW,
    0b0100011: Opcode.SW,
    0b0110111: Opcode.LUI,
    0b1100011: Opcode.BTYPE,
    0b1101111: Opcode.JAL,
    0b1100111: Opcode.JALR,
    0b1111111: Opcode.HALT,
    0b1000111: Opcode.LDM,
    0b1010111: Opcode.STM,
    0b1110111: Opcode.GEMM,
} # type: ignore