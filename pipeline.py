from opcode import Opcode, opcodes # type: ignore
from instruction import *
from funit import *
from branchpredictor import *

class PipelineStage:
    def __init__(self, name):
        self.name = name
        self.current_instruction = None

    def process(self):
        pass

class FetchStage(PipelineStage):
    def __init__(self, instruction_queue, branch_predictor):
        super().__init__("Fetch")
        self.pc = 0
        self.instruction_queue = instruction_queue
        self.branch_predictor = branch_predictor

    def process(self, tick):
        if self.pc < len(self.instruction_queue):
            self.current_instruction = self.instruction_queue[self.pc]
            self.current_instruction.pc = self.pc
            print(f"[Tick {tick}] Fetching: {self.current_instruction.opcode}")
            if self.current_instruction.opcode in {Opcode.BTYPE, Opcode.JAL, Opcode.JALR}:
                predicted_taken, used, predicted_target = self.branch_predictor.predict(self.pc)
                predicted_taken, used, predicted_target = self.branch_predictor.predict(self.pc)
                self.current_instruction.predicted_taken = predicted_taken
                self.current_instruction.predicted_target = predicted_target
                print(f"[Tick {tick}] Branch prediction: predicted_taken={predicted_taken}, target={predicted_target}, using {used} predictor.")
                if predicted_taken:
                    print(f"[Tick {tick}] Branch predicted taken; target = {predicted_target}")
                    self.current_instruction.speculative = True
                    if predicted_target is not None:
                        self.pc = predicted_target
                    else:
                        # Otherwise, compute target using your branch instruction logic.
                        self.pc = BranchUnit.compute_branch_target(self.current_instruction)
                else:
                    self.pc += 1  # Next sequential instruction.
            else:
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
            self.scoreboard.add_instruction(instruction)
            self.scoreboard.update_stage(instruction, 'D', tick)
            return instruction
        return None

class IssueStage(PipelineStage):
    def __init__(self, scoreboard):
        super().__init__("Issue")
        self.scoreboard = scoreboard

    def process(self, instruction, tick):
        if instruction is None: return None

        if instruction.opcode is Opcode.HALT:
            print(f"[Tick {tick}] Issuing: {instruction.opcode} (no FU required)")
            self.scoreboard.update_stage(instruction, 'S', tick)
            return instruction

        fu_name = None
        ins_to_fu = {Opcode.GEMM: "GEMM", Opcode.LDM: "MATLOAD", Opcode.STM: "MATLOAD", 
                     Opcode.LW: "SCALARLD", Opcode.SW: "SCALARLD", Opcode.BTYPE: "BRANCH"}
        if instruction.opcode in ins_to_fu.keys(): fu_name = ins_to_fu[instruction.opcode]
        elif instruction.opcode in opcodes.values(): fu_name = "ALU"
        else: return None
        if fu := self.scoreboard.allocate_fu(fu_name, instruction, tick):
            print(f"[Tick {tick}] Issuing: {instruction.opcode} to {fu_name} FU")
            self.scoreboard.update_stage(instruction, 'S', tick)
            instruction.fu = fu
            return instruction
        return None

class ExecuteStage(PipelineStage):
    def __init__(self, scoreboard):
        super().__init__("Execute")
        self.scoreboard = scoreboard

    def process(self, instruction, tick):
        if instruction is None: return None
        if instruction.execute():
            print(f"[Tick {tick}] Executing: {instruction.opcode}")
            self.scoreboard.update_stage(instruction, 'X', tick)
            if instruction.remaining_cycles == 0 and instruction.fu is not None:
                self.scoreboard.release_fu(instruction.fu)
            return instruction
        return None

class WriteBackStage(PipelineStage):
    def __init__(self, scoreboard, branch_predictor):
        super().__init__("WriteBack")
        self.scoreboard = scoreboard
        self.branch_predictor = branch_predictor
        self.flush_callback = None

    def process(self, instruction, tick):
        if instruction:
            # TODO: writeback logic
            print(f"[Tick {tick}] Writing back: {instruction.opcode}")
            self.scoreboard.update_stage(instruction, 'W', tick)
            if instruction.opcode in {Opcode.BTYPE, Opcode.JAL, Opcode.JALR}:
                actual_taken = instruction.taken  
                actual_target = instruction.branch_target
                predicted_taken = instruction.predicted_taken
                predicted_target = instruction.predicted_target
                self.branch_predictor.update(instruction.pc, actual_taken, actual_target)
                if predicted_taken != actual_taken or predicted_target != actual_target:
                    print(f"[Tick {tick}] Branch misprediction detected, flushing speculative instructions.")
                    if self.flush_callback:
                        self.flush_callback()
                    self.scoreboard.flush_speculative()
            return instruction
        return None