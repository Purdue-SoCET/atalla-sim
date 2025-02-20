from scoreboard import Scoreboard
from pipeline import *
from instruction import *
from funit import *

class Simulator:
    def __init__(self, decoded_instruction):
        self.tick = 0
        self.scoreboard = Scoreboard()

        dim = 4
        num_m_regs = 16
        dtype = np.float16
        
        self.scalar_regs = np.zeros(32, dtype=np.int32)
        self.matrix_regs = np.zeros((num_m_regs, dim, dim),  dtype=dtype)
        self.GEMM_unit = GEMM(dim, dtype)

        self.fetch = FetchStage(decoded_instructions)
        self.dispatch = DispatchStage(self.scoreboard)
        self.issue = IssueStage(self.scoreboard)
        self.execute = ExecuteStage(self.scoreboard)
        self.write_back = WriteBackStage(self.scoreboard)

        self.to_write_back = None
        self.to_execute = None
        self.to_issue = None
        self.to_dispatch = None
        self.fetched_instr = None

    def run(self):
        while self.tick < 15:
            print(f"\n--- Tick {self.tick} ---")

            self.write_back.process(self.to_write_back, self.tick)
            self.to_write_back = self.execute.process(self.to_execute, self.tick)
            self.to_execute = self.issue.process(self.to_issue, self.tick)
            self.to_issue = self.dispatch.process(self.fetched_instr, self.tick)
            self.fetched_instr = self.fetch.process(self.tick)
            
            self.scoreboard.print_scoreboard(self.tick)
            
            self.tick += 1


if __name__ == "__main__":
    # instruction_list = [
    #     "lw.i R1, 4(R2)",
    #     "addi.i R3, R1, 10",
    #     "sub.i R5, R4, R3",
    #     "beq.i R1, R3, 8",
    #     "jal R6, 12",
    #     "sw.i R5, 16(R1)"
    # ]
    word_list = []
    word_list.append(bytes.fromhex('20000093')[::-1]) # addi.i x1, x0, 512
    word_list.append(bytes.fromhex('1a400193')[::-1]) # addi.i x3, x0, 420
    word_list.append(bytes.fromhex('0000a103')[::-1]) # lw x2, 0(x1)
    word_list.append(bytes.fromhex('0030a223')[::-1]) # sw x3, 4(x1)
    word_list.append(bytes.fromhex('10800447')[::-1]) # ld.m m1, x0, (8)x1
    word_list.append(bytes.fromhex('31110077')[::-1]) # gemm m3, m1, m1, m1
    word_list.append(bytes.fromhex('30801457')[::-1]) # st.m m3, x0, (40)x1
    word_list.append(bytes.fromhex('FFFFFFFF')[::-1]) # HALT

    # decoded_instructions = [decode_instruction(instr) for instr in instruction_list]
    decoded_instructions = [decode_word(instr) for instr in word_list]
    sim = Simulator(decoded_instructions)
    sim.run()
