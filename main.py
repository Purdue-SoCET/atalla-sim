from scoreboard import Scoreboard
from pipeline import *
from instruction import decode_instruction

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

#DECODE LOGIC - to implement from assembler
decoded_instructions = [decode_instruction(instr) for instr in instruction_list]

fetch = FetchStage(decoded_instructions)
dispatch = DispatchStage(scoreboard)
issue = IssueStage(scoreboard)
execute = ExecuteStage(scoreboard)
write_back = WriteBackStage(scoreboard)

global_tick = 0

while global_tick < 15:
    print(f"\n--- Tick {global_tick} ---")
    
    fetched_instr = fetch.process(global_tick)
    dispatched_instr = dispatch.process(fetched_instr, global_tick)
    issued_instr = issue.process(dispatched_instr, global_tick)
    executed_instr = execute.process(issued_instr, global_tick)
    write_back.process(executed_instr, global_tick)
    
    scoreboard.print_scoreboard(global_tick)
    
    global_tick += 1