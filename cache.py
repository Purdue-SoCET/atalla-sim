import numpy as np
from copy import deepcopy
from helpers import tobits, frombits

WORD_SIZE = 32
BYTES_PER_WORD = WORD_SIZE//8
ADDR_BITS   = 32    # Address bits

CS          = 1024*2 # Cache size in bits
BLOCK_SIZE  = 2      # Block size in words
A           = 2      # Associativity
BANKS       = 4      # Number of banks
K           = 8      # MSHR size

class CacheFrame:
    def __init__(self, tag = 0, data = None, address = None, tick = -1):
        self.valid = False
        self.tag = 0
        self.address = None
        self.data = [[0 for _ in range(WORD_SIZE)] for _ in range(BLOCK_SIZE)]
        self.lru_tick = -1
    
    def __repr__(self):
        valid_str = "." if self.valid else " "
        data_str = ""
        for word_idx in range(BLOCK_SIZE):
            data_str = str(frombits(self.data[word_idx])).ljust(4, ' ') + data_str
        return valid_str + f" Tag: {self.tag}   Data: {data_str}   Addr: {hex(self.address)}   Tick: {self.lru_tick}"

class MSHREntry:
    def __init__(self, address, cache, rw_mode : int, data = None):
        self.valid = True
        self.uuid = 0
        self.write_status = BLOCK_SIZE * [0]
        self.write_block = [[0 for _ in range(WORD_SIZE)] for _ in range(BLOCK_SIZE)]
        self.add_miss(address, cache, rw_mode, data)

    def add_miss(self, address, cache, rw_mode : int, data = None):
        self.address = address
        tag, idx, blk_off, bank = cache.extract_info(address)
        # self.write_status[blk_off] = rw_mode
        # if rw_mode:
        #     self.write_block[blk_off] = data
        self.blk_off = blk_off
        self.bank = bank
        self.idx = idx
        self.tag = tag

class Cache:
    def __init__(self, dram = None):
        self.NUM_TOTAL_WORDS = (CS // 8) // BYTES_PER_WORD
        self.NUM_TOTAL_FRAMES = self.NUM_TOTAL_WORDS // BLOCK_SIZE
        self.NUM_BANKS = BANKS
        self.NUM_FRAMES_PER_BANK = self.NUM_TOTAL_FRAMES // self.NUM_BANKS
        self.NUM_SETS_PER_BANK = self.NUM_FRAMES_PER_BANK // A
        self.NUM_FRAMES_PER_SET = self.NUM_FRAMES_PER_BANK // self.NUM_SETS_PER_BANK
        print("Banks:", BANKS)
        print("-> Sets:", self.NUM_SETS_PER_BANK)
        print("   -> Frames:", self.NUM_FRAMES_PER_SET)     
        print("      -> Words:", BLOCK_SIZE)
        print("         -> Bits:", WORD_SIZE)
        
        
        # self.FRAMES = CS//(BS * WORD_SIZE)
        # self.NUM_SETS = self.FRAMES // A
        self.INDEX_BITS = int(np.log2(self.NUM_SETS_PER_BANK))
        self.BLKOFF_BITS = int(np.log2(BLOCK_SIZE))
        self.BANK_BITS = int(np.log2(self.NUM_BANKS))
        self.TAG_BITS = ADDR_BITS - self.INDEX_BITS - self.BLKOFF_BITS - self.BANK_BITS - 2

        print("Index bits     : ", self.INDEX_BITS)
        print("Block off. bits: ", self.BLKOFF_BITS)
        print("Bank idx. bits : ", self.BANK_BITS)
        print("Tag bits       : ", self.TAG_BITS)
        self.cache = [[[CacheFrame() for _ in range(A)] for _ in range(self.NUM_SETS_PER_BANK)] for _ in range(self.NUM_BANKS)]
        self.bank_latencies = self.NUM_BANKS * [0]
        self.dram = dram
        self.mshr = []
        self.data_out = {}

    def extract_info(self, address):
        address = tobits([address], bit_len=ADDR_BITS)
        # print(address)
        bank    = frombits(address[2 : 2 + self.BANK_BITS])
        blk_off = frombits(address[2 + self.BANK_BITS : 2 + self.BANK_BITS + self.BLKOFF_BITS])
        idx     = frombits(address[2 + self.BANK_BITS + self.BLKOFF_BITS: ADDR_BITS - self.TAG_BITS])
        tag     = frombits(address[ADDR_BITS - self.TAG_BITS: ])
        return tag, idx, blk_off, bank

    def main_loop(self, tick):
        status_str = f"--- Tick {tick}   MSHR: {len(self.mshr)}/{K}"
        for b_i, bank in enumerate(self.cache):
            status_str += f"  B{b_i}: +{self.bank_latencies[b_i]}".ljust(4, ' ')
        print(status_str)

        for b_i, banks in enumerate(self.cache):
            if self.bank_latencies[b_i]:
                self.bank_latencies[b_i] -= 1
        if len(self.mshr) == 0:
            return None
        entry = self.mshr[0]
        if self.bank_latencies[entry.bank] != 0:
            return None
        for frame in self.cache[entry.bank][entry.idx]:
            if frame.valid and entry.tag == frame.tag:
                data_out = deepcopy(frame.data[entry.blk_off])
                frame.lru = tick
                Cache.print_message("~ Hit      :", entry.address, entry.tag, entry.idx, entry.blk_off, entry.bank)
                self.mshr.pop(0)
                self.data_out[entry.address] = data_out
                return
        else:
            Cache.print_message("~ Miss     :", entry.address, entry.tag, entry.idx, entry.blk_off, entry.bank)
            self.mshr.pop(0)
            self.bank_latencies[entry.bank] = 20
            data_out = deepcopy(self.dram[entry.address])
            for word_idx in range(BLOCK_SIZE):
                dram_word_addr  = entry.bank << 2
                dram_word_addr += word_idx   << (2 + self.BANK_BITS)
                dram_word_addr += entry.idx  << (2 + self.BANK_BITS + self.BLKOFF_BITS)
                dram_word_addr += entry.tag  << (ADDR_BITS - self.TAG_BITS)
                data_from_dram = self.dram[dram_word_addr]
                self.write(dram_word_addr, data_from_dram, tick)
            self.data_out[entry.address] = data_out
            

    def request_read(self, address, tick):
        if len(self.mshr) == K:
            return False
        self.mshr.append(MSHREntry(address, self, 0))
        return True

    def read(self, address, tick):
        tag, idx, blk_off, bank = self.extract_info(address)
        data = None
        for frame in self.cache[bank][idx]:
            if frame.valid and tag == frame.tag:
                data = frame.data[blk_off]
                frame.lru = tick
                Cache.print_message("~ Hit      :", address, tag, idx, blk_off, bank)
                return data
        else:
            Cache.print_message("~ Miss     :", address, tag, idx, blk_off, bank)
            data = deepcopy(self.dram[address])
            for word_idx in range(BLOCK_SIZE):
                dram_word_addr  = bank     << 2
                dram_word_addr += word_idx << (2 + self.BANK_BITS)
                dram_word_addr += idx      << (2 + self.BANK_BITS + self.BLKOFF_BITS)
                dram_word_addr += tag      << (ADDR_BITS - self.TAG_BITS)
                data_from_dram = self.dram[dram_word_addr]
                self.write(dram_word_addr, data_from_dram, tick)
            return data

    def write(self, address, data, tick): 
        tag, idx, blk_off, bank = self.extract_info(address)
        cache_set = self.cache[bank][idx]
        lru = 0
        min_age = cache_set[lru].lru_tick
        for i, frame in enumerate(cache_set):
            if frame.valid and frame.tag == tag: # In place write
                frame.data[blk_off] = data
                frame.lru_tick = tick
                Cache.print_message("~ Write    :", address, tag, idx, blk_off, bank)
                return
            if frame.lru_tick < min_age : 
                min_age = frame.lru_tick
                lru = i

        lru_frame = cache_set[lru]
        if lru_frame.valid:  # Write back
            Cache.print_message("~ Write LRU:", address, tag, idx, blk_off, bank)
            for word_idx in range(BLOCK_SIZE):
                dram_word_addr  = bank          << 2
                dram_word_addr += word_idx      << (2 + self.BANK_BITS)
                dram_word_addr += idx           << (2 + self.BANK_BITS + self.BLKOFF_BITS)
                dram_word_addr += lru_frame.tag << (ADDR_BITS - self.TAG_BITS)
                self.dram[dram_word_addr] = deepcopy(lru_frame.data[word_idx])
                Cache.print_message("    -> DRAM:", dram_word_addr, lru_frame.tag, idx, word_idx, bank)
        else:
            Cache.print_message("~ Write new:", address, tag, idx, blk_off, bank)
            lru_frame.valid = True
        
        lru_frame.tag = tag
        lru_frame.address = address - blk_off * WORD_SIZE
        lru_frame.data[blk_off] = data
        lru_frame.lru_tick = tick

    @staticmethod
    def print_message(message, address, tag, idx, blkoffset, bank):
        print(f"{message} at addr. "+ str(hex(address)).ljust(5, ' ') + f" -> tag {tag} set {idx} word {blkoffset} bank {bank}")

    def __repr__(self):
        str_out = ""
        for i, bank in enumerate(self.cache):
            str_out += "Bank " + str(i) + "\n"
            for j, c_set in enumerate(self.cache[i]):
                str_out += "Set " + str(j) + "\n"
                for a in c_set:
                    str_out += str(a) + "\n"
            str_out += "\n"
        return str_out

if __name__ == "__main__":
    tick = 0
    dram = {}
    dcache = Cache(dram = dram)

    # Addressing test
    for i in range(0, dcache.NUM_TOTAL_WORDS, 4):
        tag, idx, blk_off, bank = dcache.extract_info(i)
        print(f"Addr:  {hex(i)} Tag: {int(tag)} Idx: {int(idx)} Blk_off: {blk_off} Bank {int(bank)}")

    num_test = dcache.NUM_TOTAL_WORDS * 2

    test_block_offest = True
    n = 4 if test_block_offest else 8

    print("Write test")
    for i in range(num_test):
        dcache.write(i * n, tobits([i], bit_len=WORD_SIZE), tick)
        tick += 1
    
    print("Cache after write test")
    print(dcache)

    print("DRAM after write test")
    for addr, data in sorted(dram.items()):
        print(hex(addr).ljust(5, ' '), frombits(data))

    if False:
        print("Read test")
        for i in range(num_test - 1, -1, -1):
            assert i == frombits(dcache.read(i * n, tick)), f"Incorrect read at address {hex(i * n)}"
            tick += 1
    else:
        print("Non-blocking Read test")
        i_in  = num_test - 1
        i_out = num_test - 1
        while i_out >= 0:
            if i_in >= 0:
                if dcache.request_read(i_in * n, tick):
                    i_in -= 1
            data = dcache.main_loop(tick)
            if i_out * n in dcache.data_out.keys():
                assert i_out == frombits(dcache.data_out[i_out * n]), f"Incorrect read at address {hex(i_out * n)}"
                # print("Expected:", i_out, "   out: ", frombits(dcache.data_out[i_out * n]))
                i_out -= 1
            tick += 1

    # print("Cache after read test")
    # print(dcache)

    # print("DRAM after read test")
    # for addr, data in sorted(dram.items()):
    #     print(hex(addr).ljust(5, ' '), frombits(data))
