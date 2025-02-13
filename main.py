from scoreboard import Scoreboard
from pipeline import *
from instruction import *

#Initialize
scoreboard = Scoreboard()

instruction_list = [
    "lw.i R1, 4(R2)",
    "addi.i R3, R1, 10",
    "sub.i R5, R4, R3",
    "beq.i R1, R3, 8",
    "jal R6, 12",
    "sw.i R5, 16(R1)"
]
word_list = []
word_list.append(bytes.fromhex('20000093')[::-1]) # addi.i x1, x0, 512
word_list.append(bytes.fromhex('1a400193')[::-1]) # addi.i x3, x0, 420
word_list.append(bytes.fromhex('0000a103')[::-1]) # lw x2, 0(x1)
word_list.append(bytes.fromhex('0030a223')[::-1]) # sw x3, 4(x1)
word_list.append(bytes.fromhex('10800447')[::-1]) # ld.m m1, x0, (8)x1
word_list.append(bytes.fromhex('31110077')[::-1]) # gemm m3, m1, m1, m1
word_list.append(bytes.fromhex('30801457')[::-1]) # st.m m3, x0, (40)x1
word_list.append(bytes.fromhex('FFFFFFFF')[::-1]) # HALT
print(len(word_list))

#DECODE LOGIC - to implement from assembler
# decoded_instructions = [decode_instruction(instr) for instr in instruction_list]
decoded_instructions = [decode_word(instr) for instr in word_list]


fetch = FetchStage(decoded_instructions)
dispatch = DispatchStage(scoreboard)
issue = IssueStage(scoreboard)
execute = ExecuteStage(scoreboard)
write_back = WriteBackStage(scoreboard)

global_tick = 0

to_write_back = None
to_execute = None
to_issue = None
to_dispatch = None
fetched_instr = None

while global_tick < 15:
    print(f"\n--- Tick {global_tick} ---")
    
    # fetched_instr = fetch.process(global_tick)
    # dispatched_instr = dispatch.process(fetched_instr, global_tick)
    # issued_instr = issue.process(dispatched_instr, global_tick)
    # executed_instr = execute.process(issued_instr, global_tick)
    # write_back.process(executed_instr, global_tick)

    write_back.process(to_write_back, global_tick)
    to_write_back = execute.process(to_execute, global_tick)
    to_execute = issue.process(to_issue, global_tick)
    to_issue = dispatch.process(fetched_instr, global_tick)
    fetched_instr = fetch.process(global_tick)

    
    scoreboard.print_scoreboard(global_tick)
    
    global_tick += 1