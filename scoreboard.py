from funit import *

class Scoreboard:
    def __init__(self):
        self.reg_status = {}  # Register Status Table
        self.instr_status = []  # Instruction Status Table
        self.functional_units = {"ALU": ALU(), "BRANCH": BranchUnit(), 
        "SCALARLD": ScalarLD(), "MATLOAD": MatrixLD(), "GEMM": GEMM(4, np.float16)}  # Available FUs

    def allocate_fu(self, fu_name, instruction, tick):
        fu = self.functional_units[fu_name]
        if fu is not None and fu.busy == False:
            #Check RAW hazard
           

            #Check WAW hazard
           
           
            #Allocate FU
            # fu.reset()
            self.reg_status[instruction.rd] = fu
            fu.intake(instruction)
            # self.functional_units[fu] -= 1
            return fu
        return None  #No FU available

    def release_fu(self, fu):
        # if fu in self.fu_status:
            
            
            # self.functional_units[fu] += 1  # Free FU
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

    def print_scoreboard(self, tick):
        print(f"\n[Tick {tick}] Scoreboard State:")
        print("FU Status Table:")
        for f, s in self.functional_units.items():
            print(f.ljust(8, " "), s)
        print("Register Status Table:")
        for r, s in self.reg_status.items():
            print(r, s.name)
        print("Instruction Status Table:")
        for i in self.instr_status:
            print(i)

