import numpy as np
from instruction import *
from opcode import branch_funct, alu_funct, Opcode  # type: ignore

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
        self.busy = True
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

    def compute(self, reg_file):
        a = reg_file[self.instruction.rs1]
        b = self.instruction.imm if getattr(self.instruction, 'use_imm', False) else reg_file[self.instruction.rs2]
        op = self.instruction.aluop  # This should be an AluOp value.
        self.result = alu_funct[op](a, b)
        return self.result

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
    
    def compute(self, scalar_regs, memory):
        base = scalar_regs[self.instruction.rs1]
        addr = base + self.instruction.imm
        if self.instruction.opcode == Opcode.LW:
            val = memory.load_word(addr)
            self.instruction.result = val
            return val
        elif self.instruction.opcode == Opcode.SW:
            val = scalar_regs[self.instruction.rs2]
            memory.store_word(addr, val)
            return None

class MatrixLD(FunctionalUnit):
    def __init__(self):
        super().__init__(name = "MatrixLD")

class BranchUnit(FunctionalUnit):
    def __init__(self):
        super().__init__(name = "Branch")
    
    def compute_branch(self, scalar_regs):
        if self.instruction.opcode == Opcode.JAL:
            # For JAL, the target is PC + imm and return address is PC + 4.
            taken = True
            branch_target = self.instruction.pc + self.instruction.imm
            self.instruction.return_addr = self.instruction.pc + 4
        elif self.instruction.opcode == Opcode.JALR:
            # For JALR, the target is (rs1 + imm) rounded down to an even address.
            taken = True
            branch_target = (scalar_regs[self.instruction.rs1] + self.instruction.imm) & ~1
            self.instruction.return_addr = self.instruction.pc + 4
        else:
            op1 = scalar_regs[self.instruction.rs1]
            op2 = scalar_regs[self.instruction.rs2]
            diff = op1 - op2
            # Retrieve branch operator from the instruction (a value from bfunct)
            branch_op = self.instruction.branch_cond  
            condition_func = branch_funct[branch_op]
            taken = condition_func(diff)
            # Compute branch target
            branch_target = self.instruction.pc + self.instruction.imm
        # Store the computed results in the instruction for later use
        self.instruction.taken = taken
        self.instruction.branch_target = branch_target
        return taken
    @staticmethod
    def compute_branch_target(instruction):
        return instruction.pc + instruction.imm
    @staticmethod
    def compute_jump_target(instruction):
        return instruction.pc + instruction.imm