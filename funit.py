import numpy as np
from instruction import *

class FunctionalUnit():
    def __init__(self, name = None):
        self.name = name
        self.instruction = None
        self.busy = False
        self.opcode = None
        self.rd = None
        self.rs1 = None
        self.rs2 = None
        self.t1 = None
        self.t2 = None

    def intake(self, instruction):
        self.instruction = instruction
        self.busy = self.instruction.remaining_cycles == 0
        self.opcode = instruction.opcode
        self.rd = instruction.rd
        self.rs1 = instruction.rs1
        self.rs2 = instruction.rs2
        # self.t1 = 
        # self.t2 = 

    def reset(self):
        self.instruction = None
        self.busy = False
        self.opcode = None
        self.rd = None
        self.rs1 = None
        self.rs2 = None
        self.t1 = None
        self.t2 = None
    
    def __repr__(self):
        out = ""
        out += str(self.busy).ljust(6, " ")
        if self.opcode == Opcode.ITYPE:
            opstr = str(self.instruction.aluop)[6:]
            opstr += 'I' * (self.instruction.imm != None)
        else:
            opstr = str(self.opcode)[7:]
        out += opstr.ljust(5, " ")
        out += str(self.rd).ljust(5, " ")
        out += str(self.rs1).ljust(5, " ")
        out += str(self.rs2).ljust(5, " ")
        out += str(self.t1).ljust(5, " ")
        out += str(self.t2).ljust(5, " ")
        return out

class ALU(FunctionalUnit):
    def __init__(self):
        super().__init__(name = "ALU")
        self.result = None

    def intake(self, instruction):
        super().intake(instruction)
        # op1 = self.scalar_regs[i.rs1]
        # op2 =  i.imm if i.use_imm else self.scalar_regs[i.rs2]
        # out = alu_funct[i.aluop](op1,op2)
        # return out
        

class GEMM(FunctionalUnit):
    def __init__(self, dim : int, dtype : np.dtype):
        super().__init__(name = "GEMM")
        self.dim = dim
        self.dtype = dtype
        self.w = np.zeros((dim, dim), dtype)
        mac_latency = self.w.dtype.itemsize * 8
        self.latency = 2 * dim - 1 # Shifting 
        self.latency += (2 * dim - 1) * mac_latency # Computation
        self.most_recent_weight = None
        self.weights_changed = False
    
    def intake(self, instruction):
        super().intake(instruction)
        instruction.latency = self.latency
        if instruction.ra != self.most_recent_weight or self.weights_changed:
            instruction.latency += self.dim
            # self.w = 
        # self.i = 
        # self.a = 
        # result = np.matmul(self.w, self.i) + self.a


class ScalarLD(FunctionalUnit):
    def __init__(self):
        super().__init__(name = "ScalarLD")

class MatrixLD(FunctionalUnit):
    def __init__(self):
        super().__init__(name = "MatrixLD")

class BranchUnit(FunctionalUnit):
    def __init__(self):
        super().__init__(name = "Branch")
