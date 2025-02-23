from helpers import *
from opcode import Opcode, opcodes, AluOp, BranchOp, rfunct, ifunct, bfunct# type: ignore
#add decoding logic for instruction

class Instruction:
    def __init__(self, opcode, dest=None, src1=None, src2=None, latency=1):
        self.opcode = opcode
        self.fu = None
        self.rd = dest
        self.rs1 = src1
        self.rs2 = src2
        self.ra = None
        self.rb = None
        self.rc = None
        self.imm = None
        self.latency = latency
        self.remaining_cycles = latency

    def execute(self):
        if self.remaining_cycles > 0:
            self.remaining_cycles -= 1
        return (self.remaining_cycles == 0)  #execution done

    def __repr__(self):
        st = ""
        if self.opcode is Opcode.HALT: return (st + "HALT").ljust(24, ' ')
        if self.aluop: 
            if self.opcode is Opcode.BTYPE:
                st += str(self.branch_cond)[9:].lower()
            else:
                if self.opcode in {Opcode.SW, Opcode.LW, Opcode.LUI, 
                                    Opcode.LDM, Opcode.STM, Opcode.GEMM}:
                    st += str(self.opcode)[7:].ljust(4, ' ').lower()
                else:
                    st += (str(self.aluop).lower()[6:] + 'i'*self.use_imm).ljust(4, ' ')
        var = ", x"
        if self.opcode in {Opcode.STM, Opcode.LDM, Opcode.GEMM}: 
            var = ", m"
        if self.rd is not None:     st += var[1:] + str(self.rd)
        if self.rs1 is not None and self.opcode not in {Opcode.SW, Opcode.LW}: st += ", x" + str(self.rs1) 
        if self.rs2 is not None:    st += var + str(self.rs2) 
        if self.ra is not None:     st += var + str(self.ra) 
        if self.rb is not None:     st += var + str(self.rb)
        if self.rc is not None:     st += var + str(self.rc) 
        if self.imm is not None:
            if self.opcode not in {Opcode.SW, Opcode.LW, Opcode.LDM, Opcode.STM}:
                st += var[:-1] + str(self.imm)
            else:
                st += f", {self.imm}(x{self.rs1})"
        if self.opcode is Opcode.BTYPE: st = st[0:5] + ' ' + st[6:]
        if self.opcode is Opcode.SW: st = st[0:6] + st[7:]
        return st.ljust(24, " ")

def decode_instruction(instr_str):
    #decode logic)
    print(instr_str)

def decode_word(instruction: bytes):
    instruction = instruction[::-1]
    # print(instruction.hex())
    assert len(instruction) == 4, "instructions should be four bytes"
    bits = tobits(instruction)
    # print(bits)
    def bit_range(a, b = None):
        if b is None: b = a
        return bits[a: b + 1]

    opcode = opcodes[frombits(bit_range(0,6))]
    instr = Instruction(opcode=opcode)
    if opcode is Opcode.HALT:
        return instr
    if opcode is Opcode.RTYPE or opcode is Opcode.MUL:
        instr.rd = frombits(bit_range(7,11))
        instr.rs1 = frombits(bit_range(15,19))
        instr.rs2 = frombits(bit_range(20,24))
        # funct7 ++ funct3
        instr.aluop = rfunct[frombits(bit_range(25,31) + bit_range(12,14))]
        return instr
    if opcode is Opcode.ITYPE or opcode is Opcode.LW:
        instr.use_imm = True
        instr.rd = frombits(bit_range(7,11))
        instr.rs1 = frombits(bit_range(15,19))
        instr.imm = frombits(bit_range(20,31), signed = True)
        instr.aluop = ifunct[frombits(bit_range(12,14))]
        # if instr.aluop is AluOp.SRL and frombits(imm[5:11]) == 0x20: instr.aluop = AluOp.SRA
        return instr
    if opcode is Opcode.SW:
        instr.use_imm = True
        instr.imm = frombits(bit_range(7, 11) + bit_range(25,31))
        instr.rs1 = frombits(bit_range(15,19))
        instr.rs2 = frombits(bit_range(20,24))
        instr.aluop = ifunct[frombits(bit_range(12,14))]
        return instr
    if opcode is Opcode.BTYPE:
        instr.use_imm = False
        imm = ([0] + bit_range(8,11) + bit_range(25,30) + [bits[7]] + [bits[31]])
        instr.imm = frombits(imm, signed = True)
        instr.rs1 = frombits(bit_range(15,19))
        instr.rs2 = frombits(bit_range(20,24))
        instr.aluop = AluOp.SUB # resolve sign extensions in decode
        instr.branch_cond = bfunct[frombits(bit_range(12,14))]
        return instr
    if opcode is Opcode.JAL:
        instr.aluop = AluOp.NOP
        instr.rd = frombits(bit_range(7,11))
        instr.imm = frombits([0] + bit_range(21,30) + [bits[20]] + bit_range(12,19) + [bits[31]])
        return instr
    if opcode is Opcode.LUI:
        instr.aluop = AluOp.NOP
        instr.rd = frombits(bit_range(7,11))
        instr.imm = frombits(bit_range(12,31))
        return instr
    if opcode is Opcode.GEMM:
        instr.rc = frombits(bit_range(16,19))
        instr.rb = frombits(bit_range(20,23))
        instr.ra = frombits(bit_range(24,27))
        instr.rd = frombits(bit_range(28,31))
        instr.aluop = AluOp.NOP
        return instr
    if opcode is Opcode.STM or opcode is Opcode.LDM:
        instr.use_imm = True
        instr.rd = frombits(bit_range(28, 31))
        instr.rs1 = frombits(bit_range(23, 27))
        instr.stride = frombits(bit_range(18, 22))
        instr.imm = frombits(bit_range(7, 17))
        instr.aluop = AluOp.ADD
        return instr
    assert False, f"malformed instruction: f{bits}"

