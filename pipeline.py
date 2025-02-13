from opcode import *

class PipelineStage:
    def __init__(self, name):
        self.name = name
        self.current_instruction = None

    def process(self):
        
        pass

class FetchStage(PipelineStage):
    def __init__(self, instruction_queue):
        super().__init__("Fetch")
        self.pc = 0
        self.instruction_queue = instruction_queue

    def process(self, tick):
        if self.pc < len(self.instruction_queue):
            self.current_instruction = self.instruction_queue[self.pc]
            print(f"[Tick {tick}] Fetching: {self.current_instruction.opcode}")
            self.pc += 1
            return self.current_instruction
        return None

class DispatchStage(PipelineStage):
    def __init__(self, scoreboard):
        super().__init__("Dispatch")
        self.scoreboard = scoreboard

    def process(self, instruction, tick):
        if instruction:
            print(f"[Tick {tick}] Dispatching: {instruction.opcode}")
            self.scoreboard.update_stage(instruction, 'D', tick)
            self.scoreboard.add_instruction(instruction)
            return instruction
        return None

class IssueStage(PipelineStage):
    def __init__(self, scoreboard):
        super().__init__("Issue")
        self.scoreboard = scoreboard

    def process(self, instruction, tick):
        if instruction is None: return None
        #placeholder (add issue queue logic)

        # TODO: Check RAW hazard
        fu = None
        ins_to_fu = {Opcode.GEMM: "GEMM", Opcode.LDM: "MATLOAD", Opcode.STM: "MATLOAD", 
                     Opcode.LW: "SCALARLD", Opcode.SW: "SCALARLD", Opcode.BTYPE: "BRANCH"}
        if instruction.opcode in ins_to_fu.keys(): fu = ins_to_fu[instruction.opcode]
        elif instruction.opcode in opcodes.values(): fu = "ALU"
        else: return None
        if self.scoreboard.allocate_fu(fu, instruction, tick):
            print(f"[Tick {tick}] Issuing: {instruction.opcode} to {fu} FU")
            return instruction
        return None

class ExecuteStage(PipelineStage):
    def __init__(self, scoreboard):
        super().__init__("Execute")
        self.scoreboard = scoreboard

    def process(self, instruction, tick):
        if instruction and instruction.execute():
            print(f"[Tick {tick}] Executing: {instruction.opcode}")
            self.scoreboard.update_stage(instruction, 'X', tick)
            return instruction
        return None

class WriteBackStage(PipelineStage):
    def __init__(self, scoreboard):
        super().__init__("WriteBack")
        self.scoreboard = scoreboard

    def process(self, instruction, tick):
        if instruction:
            print(f"[Tick {tick}] Writing back: {instruction.opcode}")
            self.scoreboard.update_stage(instruction, 'W', tick)
            self.scoreboard.release_fu("ALU")