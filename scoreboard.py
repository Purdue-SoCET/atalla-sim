from funit import *

class Scoreboard:
    def __init__(self):
        self.reg_status = {}  # Register Status Table
        self.matrix_status = {}
        self.instr_status = []  # Instruction Status Table
        self.functional_units = {"ALU": ALU(), "BRANCH": BranchUnit(), 
        "SCALARLD": ScalarLD(), "MATLOAD": MatrixLD(), "GEMM": GEMM(4, np.float16)}  # Available FUs

    def allocate_fu(self, fu_name, instruction, tick):
        fu = self.functional_units[fu_name]
        if fu is not None and fu.busy == False:
            is_matrix = fu_name in ("MATLOAD", "GEMM")
            #Check RAW hazard
            if instruction.rd is not None and instruction.opcode != Opcode.LW:
                if ((self.reg_status.get(instruction.rs1, None) and self.reg_status[instruction.rs1] != fu) or
                    (self.reg_status.get(instruction.rs2, None) and self.reg_status[instruction.rs2] != fu)):
                        print(f"[Tick {tick}] RAW Hazard: Cannot issue {instruction.opcode}")
                        return False

            #Check WAW hazard
            if instruction.rd is not None:
                if is_matrix:
                    if instruction.rd in self.matrix_status:
                        print(f"[Tick {tick}] WAW Hazard: Cannot issue {instruction.opcode}")
                        return False
                else:
                    if instruction.rd in self.reg_status:
                        print(f"[Tick {tick}] WAW Hazard: Cannot issue {instruction.opcode}")
                        return False
           
            #Allocate FU
            # fu.reset()
            if instruction.rd is not None:
                if is_matrix:
                    self.matrix_status[instruction.rd] = fu
                else:
                    self.reg_status[instruction.rd] = fu
            fu.intake(instruction)
            # self.functional_units[fu] -= 1
            return fu
        return None  #No FU available

    def release_fu(self, fu):
        # if fu in self.fu_status:
        if fu.rd is not None:
            if fu.name in ("MATLOAD", "GEMM"):
                if fu.rd in self.matrix_status:
                    del self.matrix_status[fu.rd]
            else:
                if fu.rd in self.reg_status:
                    del self.reg_status[fu.rd]
        fu.reset()

    def add_instruction(self, instruction):
        #Track an instruction in the scoreboard
        self.instr_status.append({'instr': instruction, 'D': 0, 'S': 0, 'X': 0, 'W': 0})

    def update_stage(self, instruction, stage, tick):
        for ins in self.instr_status:
            if ins['instr'] == instruction:
                ins[stage] = tick
                break

    def flush_speculative(self):
        flushed = [ins for ins in self.instr_status if ins['instr'].speculative]
        self.instr_status = [ins for ins in self.instr_status if not ins['instr'].speculative]
        print(f"Flushed {len(flushed)} speculative instructions from scoreboard.")

    def print_scoreboard(self, tick):
        print(f"\n[Tick {tick}] Scoreboard State:")
        print("FU Status Table:")
        for f, s in self.functional_units.items():
            print(f.ljust(8, " "), s)
        print("Scalar Register Status Table:")
        for r, s in self.reg_status.items():
            print(r, s.name)
        print("Matrix Register Status Table:")
        for r, s in self.matrix_status.items():
            print(r, s.name)
        print("Instruction Status Table:")
        for i in self.instr_status[:-20:-1][::-1]:
            print(i)

