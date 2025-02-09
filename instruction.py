
#add decoding logic for instruction

class Instruction:
    def __init__(self, opcode, dest=None, src1=None, src2=None, latency=1):
        self.opcode = opcode
        self.dest = dest
        self.src1 = src1
        self.src2 = src2
        self.latency = latency
        self.remaining_cycles = latency

    def execute(self):
        if self.remaining_cycles > 0:
            self.remaining_cycles -= 1
        return (self.remaining_cycles == 0)  #execution done

def decode_instruction(instr_str):
    #decode logic)
    print(instr_str)