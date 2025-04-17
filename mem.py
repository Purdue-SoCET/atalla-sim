class Memory:
    def __init__(self, size=4096):
        self.mem = bytearray(size)
        self.size = size

    def load_word(self, addr):
        if addr % 4 != 0:
            raise ValueError(f"Unaligned memory access at address {addr:08x}")
        if addr + 4 > self.size:
            raise ValueError("Memory read out of bounds")
        return int.from_bytes(self.mem[addr:addr + 4], byteorder='little')

    def store_word(self, addr, value):
        if addr % 4 != 0:
            raise ValueError(f"Unaligned memory access at address {addr:08x}")
        if addr + 4 > self.size:
            raise ValueError("Memory write out of bounds")
        self.mem[addr:addr + 4] = value.tobytes()

    def dump(self, start=0, end=None):
        if end is None:
            end = self.size
        print("==== MEMORY DUMP ====")
        for addr in range(start, end, 4):
            word = self.load_word(addr)
            if word != 0:
                print(f"0x{addr:08x}: 0x{word:08x}")