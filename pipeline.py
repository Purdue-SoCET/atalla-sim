from opcode import Opcode, opcodes # type: ignore
from instruction import *
from funit import *
from branchpredictor import *
from cache import *

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
        self.i_mem = {}
        for addr, inst in enumerate(instruction_queue):
            self.i_mem[addr] = np.frombuffer(inst, dtype=np.uint32)
        self.icache = Cache(size=2048, block_size=4, assoc=2, banks=4, mshr_k=8, dram=self.i_mem)
        self.icache.DRAM_LATENCY = 2
        self.branch_predictor = branch_predictor

    def process(self, tick):
        if self.pc < len(self.instruction_queue):
            # self.current_instruction = self.instruction_queue[self.pc]
            stalling = False
            for bank_busy in self.icache.bank_current_mshr:
                if bank_busy: stalling = True
            if stalling:
                self.current_instruction = None
                print("Icache stall")
                return self.current_instruction
            else:
                request_result = self.icache.request(address = self.pc, is_write = False, data_in = None, tick=tick)
                not_hit = isinstance(request_result, str)
            if not_hit:
                self.current_instruction = None
                print("Instruction miss at", hex(self.pc), request_result)
                return self.current_instruction
            else:
                self.current_instruction = decode_word(request_result.tobytes())
            self.current_instruction.pc = self.pc
            print(f"[Tick {tick}] Fetching: {self.current_instruction.opcode}")
            if self.current_instruction.opcode == Opcode.BTYPE:
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
            elif self.current_instruction.opcode in {Opcode.JAL, Opcode.JALR}:
                self.current_instruction.predicted_taken = True
                if self.current_instruction.opcode == Opcode.JAL:
                    target = BranchUnit.compute_jump_target(self.current_instruction)
                else:
                    target = self.current_instruction.pc + self.current_instruction.imm
                self.current_instruction.predicted_target = target
                self.current_instruction.speculative = True
                print(f"[Tick {tick}] Jump instruction: {self.current_instruction.opcode} target={target}")
                self.pc = target
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
        else: return instruction
        fu = self.scoreboard.allocate_fu(fu_name, instruction, tick)
        if fu:
            print(f"[Tick {tick}] Issuing: {instruction.opcode} to {fu_name} FU")
            self.scoreboard.update_stage(instruction, 'S', tick)
            instruction.fu = fu
            return instruction
        else:
            print(f"[Tick {tick}] Stalling: {instruction.opcode} waiting for {fu_name} FU")
            return instruction

class ExecuteStage(PipelineStage):
    def __init__(self, scoreboard, scalar_regs):
        super().__init__("Execute")
        self.scoreboard = scoreboard
        self.scalar_regs = scalar_regs

    def process(self, instruction, tick):
        if instruction is None: return None
        if instruction.opcode == Opcode.HALT:
            finished = True
        elif instruction.opcode in {Opcode.LW, Opcode.SW}:
            request_result = self.scoreboard.functional_units["SCALARLD"].compute(self.scalar_regs, tick)
            finished = not isinstance(request_result, str)
        else:
            finished = instruction.execute()
        self.scoreboard.update_stage(instruction, 'X', tick)
        if finished:
            instruction.remaining_cycles = 0
            print(f"[Tick {tick}] Executing complete: {instruction.opcode}")
            if instruction.opcode in {Opcode.ITYPE, Opcode.RTYPE} and instruction.opcode not in {Opcode.LW, Opcode.SW, Opcode.LDM, Opcode.STM}:
                if instruction.fu is not None:
                    result = instruction.fu.compute(self.scalar_regs, tick)
                    instruction.result = result
                    print(f"[Tick {tick}] Computed result: {result}")
            if instruction.fu is not None:
                self.scoreboard.release_fu(instruction.fu)
            return instruction
        else:
            print(f"[Tick {tick}] Executing (stalled): {instruction.opcode}, remaining_cycles={instruction.remaining_cycles}")
            return instruction

class WriteBackStage(PipelineStage):
    def __init__(self, scoreboard, branch_predictor, scalar_regs):
        super().__init__("WriteBack")
        self.scoreboard = scoreboard
        self.branch_predictor = branch_predictor
        self.flush_callback = None
        self.scalar_regs = scalar_regs

    def process(self, instruction, tick):
        if instruction:
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
                else:
                    self.scoreboard.flush_speculative()
            elif instruction.opcode in {Opcode.ITYPE, Opcode.RTYPE} and instruction.opcode not in {Opcode.LW, Opcode.SW, Opcode.LDM, Opcode.STM}:
                if instruction.fu is not None:
                    result = instruction.result
                    if instruction.rd is not None:
                        self.scalar_regs[instruction.rd] = result
                    print(f"[Tick {tick}] ALU result: {result} written to x{instruction.rd}")
            # if instruction.fu is not None:
            #     self.scoreboard.release_fu(instruction.fu)
            return instruction
        return None