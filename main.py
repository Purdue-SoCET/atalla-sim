import sys
import numpy as np
from scoreboard import Scoreboard
from pipeline import *
from instruction import *
from funit import *
from branchpredictor import Tournament
from helpers import load_instructions
from mem import Memory

class Simulator:
    def __init__(self, decoded_instruction):
        self.tick = 0
        self.scoreboard = Scoreboard()

        self.branch_predictor = Tournament(num_entries=1024, ghr_bits=10)
        self.memory = Memory(4096)
        
        dim = 4
        num_m_regs = 16
        dtype = np.float16
        
        self.scalar_regs = np.zeros(32, dtype=np.int32)
        self.matrix_regs = np.zeros((num_m_regs, dim, dim),  dtype=dtype)
        self.GEMM_unit = GEMM(dim, dtype)

        self.fetch = FetchStage(decoded_instructions, self.branch_predictor)
        self.dispatch = DispatchStage(self.scoreboard)
        self.issue = IssueStage(self.scoreboard)
        self.execute = ExecuteStage(self.scoreboard, self.scalar_regs, self.memory)
        self.write_back = WriteBackStage(self.scoreboard, self.branch_predictor, self.scalar_regs)
        self.write_back.flush_callback = self.flush_pipeline

        self.to_write_back = None
        self.to_execute = None
        self.to_issue = None
        self.to_dispatch = None
        self.fetched_instr = None

        self.writeback_queue = []
        self.execute_queue = []
        self.issue_queue = []

    def flush_pipeline(self):
        print(f"[Tick {self.tick}] Flushing pipeline registers due to branch misprediction.")
        if self.to_write_back and self.to_write_back.speculative:
            self.to_write_back = None
        if self.to_execute and self.to_execute.speculative:
            self.to_execute = None
        if self.to_issue and self.to_issue.speculative:
            self.to_issue = None
        if self.fetched_instr and self.fetched_instr.speculative:
            self.fetched_instr = None

    def run(self):
        while self.tick < 15:
            print(f"\n--- Tick {self.tick} ---")

            # wb_instr = self.write_back.process(self.to_write_back, self.tick) if self.to_write_back else None
            # self.to_write_back = None
            # exe_instr = self.execute.process(self.to_execute, self.tick) if self.to_execute else None
            # issue_instr = self.issue.process(self.to_issue, self.tick) if self.to_issue else None
            # dispatch_instr = self.dispatch.process(self.fetched_instr, self.tick) if self.fetched_instr else None
            # fetch_instr = self.fetch.process(self.tick)

            # if self.to_execute is not None and self.to_execute.remaining_cycles == 0:
            #     self.to_write_back = self.to_execute
            #     self.to_execute = None
            # else:
            #     if exe_instr is not None:
            #         self.to_execute = exe_instr

            # if self.to_execute is None and self.to_issue is not None:
            #     if self.to_issue.opcode is Opcode.HALT or self.to_issue.fu is not None:
            #         self.to_execute = self.to_issue
            #         self.to_issue = None
            #     else:
            #         self.to_issue = issue_instr if issue_instr is not None else self.to_issue

            # if self.to_issue is None: 
            #     if self.to_dispatch is not None:
            #         self.to_issue = self.to_dispatch
            #         self.to_dispatch = None
            #     else:
            #         if dispatch_instr is not None:
            #             self.to_issue = dispatch_instr
            # else:
            #     if dispatch_instr is not None and self.to_dispatch is None:
            #         self.to_dispatch = dispatch_instr
            
            # self.fetched_instr = fetch_instr

            # WriteBack Phase: process all instructions in writeback queue.
            new_wb = []
            for instr in self.writeback_queue:
                self.write_back.process(instr, self.tick)
            self.writeback_queue = new_wb  # clear WB queue
            
            # Execute Phase: process all instructions in execute queue.
            new_exec = []
            for instr in self.execute_queue:
                result = self.execute.process(instr, self.tick)
                if result and result.remaining_cycles == 0:
                    # Release the FU immediately, and push into writeback queue.
                    self.writeback_queue.append(result)
                else:
                    new_exec.append(instr)
            self.execute_queue = new_exec
            
            # Issue stage (only one instruction deep).
            if self.to_issue is not None:
                issued_instr = self.issue.process(self.to_issue, self.tick)
                # If the instruction is ready for execution (e.g. HALT or FU allocated), 
                # promote it to Execute list.
                if self.to_issue.opcode == Opcode.HALT or self.to_issue.fu is not None:
                    self.execute_queue.append(self.to_issue)
                    self.to_issue = None
                else:
                    self.to_issue = issued_instr  # remains stalled
            
            # Dispatch stage.
            if self.to_issue is None and self.fetched_instr is not None:
                disp_instr = self.dispatch.process(self.fetched_instr, self.tick)
                if disp_instr is not None:
                    self.to_issue = disp_instr
                    self.fetched_instr = None
            # (If issue stage is full, the fetched instruction remains in self.fetched_instr.)
            
            # Fetch stage.
            # Only fetch a new instruction if the fetch register is empty.
            if self.fetched_instr is None:
                self.fetched_instr = self.fetch.process(self.tick)

            self.scoreboard.print_scoreboard(self.tick)
            
            self.tick += 1

    def dump_registers(self):
        print("\n=== Final Scalar Registers ===")
        for i, reg in enumerate(self.scalar_regs):
            print(f"x{i:2d}: {reg}")
        print("\n=== Final Matrix Registers ===")
        for i, mat in enumerate(self.matrix_regs):
            print(f"m{i}:")
            print(mat)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        binary_file = sys.argv[1]
        word_list = load_instructions(binary_file)
    else:
        word_list = []
        word_list.append(bytes.fromhex('20000093')[::-1]) # addi.i x1, x0, 512
        word_list.append(bytes.fromhex('1a400193')[::-1]) # addi.i x3, x0, 420
        word_list.append(bytes.fromhex('0000a103')[::-1]) # lw x2, 0(x1)
        word_list.append(bytes.fromhex('0030a223')[::-1]) # sw x3, 4(x1)
        word_list.append(bytes.fromhex('01010263')[::-1])  # beq x1, x2, 16   (example encoding)
        word_list.append(bytes.fromhex('F0E1C263')[::-1])  # bne x1, x3, -16  (example encoding)
        word_list.append(bytes.fromhex('10800447')[::-1]) # ld.m m1, x0, (8)x1
        word_list.append(bytes.fromhex('31110077')[::-1]) # gemm m3, m1, m1, m1
        word_list.append(bytes.fromhex('30801457')[::-1]) # st.m m3, x0, (40)x1
        word_list.append(bytes.fromhex('FFFFFFFF')[::-1]) # HALT

    # decoded_instructions = [decode_instruction(instr) for instr in instruction_list]
    decoded_instructions = [decode_word(instr) for instr in word_list]
    sim = Simulator(decoded_instructions)
    sim.run()
    sim.dump_registers()
    Memory.dump(sim.memory)
