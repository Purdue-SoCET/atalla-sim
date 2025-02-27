import numpy as np
from copy import deepcopy
from helpers import tobits, frombits

WORD_SIZE = 32
BYTES_PER_WORD = WORD_SIZE//8

CS = 1024        # Cache size in bits
BS = 2           # Block size in words
A = 2            # Associativity
ADDR_BITS = 32   # Address bits

# 8 sets of 2 frames
# Each block has 2 words, each word is 4 bytes

class CacheFrame:
    def __init__(self, tag = 0, data = None, address = None, tick = -1):
        self.valid = False
        self.tag = 0
        self.address = None
        self.data = [0 for _ in range(BS * WORD_SIZE)]
        self.lru_tick = -1
    
    def __repr__(self):
        valid_str = "." if self.valid else " "
        data_str = ""
        for word_idx in range(BS):
            data_str = str(frombits(self.data[word_idx * WORD_SIZE : (word_idx + 1) * WORD_SIZE])) + "   " + data_str
        return valid_str + f" Tag: {self.tag}   Data: {data_str}   Addr: {self.address}   Tick: {self.lru_tick}"

class Cache:
    def __init__(self):
        self.FRAMES = CS//(BS * WORD_SIZE)
        self.NUM_SETS = self.FRAMES // A
        self.INDEX_BITS = int(np.log2(self.NUM_SETS));         # Currently: 3 bits
        self.BLKOFF_BITS = int(np.log2(BS));              # Currently: 1 bit
        self.BYTEOFF_BITS = int(np.log2(BYTES_PER_WORD)); # Currently: 2 bits
        self.TAG_BITS = ADDR_BITS - self.INDEX_BITS - self.BLKOFF_BITS - self.BYTEOFF_BITS; #Currently: 26 bits
        # print(self.FRAMES)
        # print(self.NUM_SETS)
        # print(self.INDEX_BITS)
        # print(self.BLKOFF_BITS)
        # print(self.BYTEOFF_BITS)
        # print(self.TAG_BITS)
        self.cache = [[CacheFrame() for _ in range(A)] for _ in range(self.NUM_SETS)]
        self.dram = None
        print(f"Cache created: {self.NUM_SETS} sets of assoc. {A}")

    def extract_info(self, address):
        address = tobits([address], bit_len=ADDR_BITS)
        # print(address)
        tag    = frombits(address[ADDR_BITS - self.TAG_BITS : ADDR_BITS])
        idx    = frombits(address[self.BYTEOFF_BITS + self.BLKOFF_BITS : ADDR_BITS - self.TAG_BITS])
        blkoff = frombits(address[self.BYTEOFF_BITS + self.BLKOFF_BITS - 1])
        bytoff = frombits(address[0 : self.BYTEOFF_BITS])
        # print("~~~~~~~~", blkoff)
        return tag, idx, blkoff, bytoff

    def read(self, address, tick):
        tag, idx, blkoff, bytoff = self.extract_info(address)
        data = None
        for frame in self.cache[idx]:
            if frame.valid and tag == frame.tag:
                data_idx = WORD_SIZE * blkoff
                data = frame.data[data_idx : data_idx + WORD_SIZE]
                frame.lru = tick
                Cache.print_message("~ Hit      :", address, tag, idx, blkoff)
                return data
        else:
            Cache.print_message("~ Miss     :", address, tag, idx, blkoff)
            base_addr = address - BYTES_PER_WORD * blkoff
            for word_idx in range(BS):  # Write missed block to cache
                dram_word_addr = base_addr + word_idx * BYTES_PER_WORD
                data = deepcopy(self.dram[dram_word_addr])
                self.write(dram_word_addr, data, tick)
            return deepcopy(self.dram[address])

    def write(self, address, data, tick): 
        tag, idx, blkoff, bytoff = self.extract_info(address)
        cache_set = self.cache[idx]
        lru = 0
        min_age = cache_set[lru].lru_tick
        for i, frame in enumerate(cache_set):
            if frame.valid and frame.tag == tag: # In place write
                data_idx = WORD_SIZE * blkoff
                frame.data[data_idx : data_idx + WORD_SIZE] = data
                frame.lru_tick = tick
                Cache.print_message("~ Write    :", address, tag, idx, blkoff)
                return
            if frame.lru_tick < min_age : 
                min_age = frame.lru_tick
                lru = i

        lru_frame = cache_set[lru]
        if lru_frame.valid:  # Write back
            Cache.print_message("~ Write LRU:", address, tag, idx, blkoff)
            for word_idx in range(BS):
                dram_word_addr = lru_frame.address + word_idx * BYTES_PER_WORD
                self.dram[dram_word_addr] = deepcopy(lru_frame.data[word_idx * WORD_SIZE : (word_idx + 1) * WORD_SIZE])
                Cache.print_message("    -> DRAM:", dram_word_addr, lru_frame.tag, lru, word_idx)
        else:
            Cache.print_message("~ Write new:", address, tag, idx, blkoff)
            lru_frame.valid = True
        
        lru_frame.tag = tag
        lru_frame.address = address - blkoff * WORD_SIZE
        idx = WORD_SIZE * blkoff
        lru_frame.data[idx : idx + WORD_SIZE] = data
        lru_frame.lru_tick = tick

    @staticmethod
    def print_message(message, address, tag, idx, blkoffset):
        print(f"{message} at addr. "+ str(address).ljust(5, ' ') + f" -> set {idx} tag {tag} blk {blkoffset}")

    def __repr__(self):
        str_out = ""
        for i, set in enumerate(self.cache):
            str_out += "Set " + str(i) + "\n"
            for a in set:
                str_out += str(a) + "\n"
        return str_out

if __name__ == "__main__":
    tick = 0
    dram = {}
    dcache = Cache()
    dcache.dram = dram
    num_test = dcache.FRAMES * 4

    test_block_offest = True
    n = 4 if test_block_offest else 8

    print("Write test")
    for i in range(num_test):
        dcache.write(i * n, tobits([i], bit_len=WORD_SIZE), tick)
        tick += 1
    
    # In place write
    dcache.write((num_test - 1) * n, tobits([num_test - 1], bit_len=WORD_SIZE), tick)
    tick += 1
    
    print("Cache after write test")
    print(dcache)

    print("DRAM after write test")
    for addr, data in sorted(dram.items()):
        print(addr, frombits(data))

    print("Read test")
    for i in range(num_test):
        assert i == frombits(dcache.read(i * n, tick)), f"Incorrect read at address {i * n}"
        tick += 1

    print("Cache after read test")
    print(dcache)

    print("DRAM after read test")
    for addr, data in sorted(dram.items()):
        print(addr, frombits(data))
