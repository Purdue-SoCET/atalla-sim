class Scoreboard:
    def __init__(self):
        self.fu_status = { # FU Status Table
            "ALU": {'busy': False, 'op': None, 'R': None, 'R1': None, 'R2': None, 'T1': None, 'T2': None},
            "BRANCH": {'busy': False, 'op': None, 'R': None, 'R1': None, 'R2': None, 'T1': None, 'T2': None},
            "SCALARLD": {'busy': False, 'op': None, 'R': None, 'R1': None, 'R2': None, 'T1': None, 'T2': None},
            "MATLOAD": {'busy': False, 'op': None, 'R': None, 'R1': None, 'R2': None, 'T1': None, 'T2': None},
            "GEMM": {'busy': False, 'op': None, 'R': None, 'R1': None, 'R2': None, 'T1': None, 'T2': None}
        }  
        self.reg_status = {}  # Register Status Table
        self.instr_status = []  # Instruction Status Table
        self.functional_units = {"ALU": 1, "BRANCH": 1, "SCALARLD": 1, "MATLOAD": 1, "GEMM": 1}  # Available FUs

    def allocate_fu(self, fu, instruction, tick):
        if self.functional_units.get(fu, 0) > 0:
            #Check RAW hazard
           

            #Check WAW hazard
           
           
            #Allocate FU
            self.fu_status[fu] = {
                'busy': True,
                'op': instruction.opcode,
                'R': instruction.rd,
                'R1': instruction.rs1,
                'R2': instruction.rs2,
                'T1': self.reg_status.get(instruction.rs1, 0),
                'T2': self.reg_status.get(instruction.rs2, 0)
            }
            self.reg_status[instruction.rd] = fu
            self.functional_units[fu] -= 1
            return True
        return False  #No FU available

    def release_fu(self, fu):
        # if fu in self.fu_status:
            
            
            self.functional_units[fu] += 1  # Free FU
            self.fu_status[fu] = {'busy': False, 'op': None, 'R': None, 'R1': None, 'R2': None, 'T1': None, 'T2': None}

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
        for f, s in self.fu_status.items():
            print(f, s)
        print("Register Status Table:")
        for r, s in self.reg_status.items():
            print(r, s)
        print("Instruction Status Table:")
        for i in self.instr_status:
            print(i)

