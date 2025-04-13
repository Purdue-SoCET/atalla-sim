import numpy as np
from copy import deepcopy
from helpers import tobits, frombits

WORD_SIZE = 32
BYTES_PER_WORD = WORD_SIZE//8
ADDR_BITS   = 32    # Address bits

DRAM_LATENCY = 20

class CacheFrame:
    def __init__(self, block_size = 4, tag = 0, data = None, address = None, tick = -1):
        self.block_size = block_size
        self.valid = False
        self.dirty = False
        self.tag = 0
        self.address = 0
        self.data = [[0 for _ in range(WORD_SIZE)] for _ in range(block_size)]
        self.lru_tick = -1
    
    def __repr__(self):
        valid_str = "." if self.valid else " "
        data_str = ""
        for word_idx in range(self.block_size):
            data_str = str(frombits(self.data[word_idx])).ljust(4, ' ') + data_str
        return valid_str + f" Tag: {self.tag}   Data: {data_str}   Addr: {hex(self.address)}   Tick: {self.lru_tick}"

class MSHRMiss:
    def __init__(self, address, blk_off, is_write, data_in, uuid):
        self.valid = True
        self.address = address
        self.blk_off = blk_off
        self.is_write = is_write
        self.data_in = data_in
        self.uuid = uuid

class MSHREntry:
    def __init__(self, address, cache, is_write, data_in, uuid):
        self.address = address
        self.misses = []
        self.add_miss(address, cache, is_write, data_in, uuid)

    def add_miss(self, address, cache, is_write, data_in, uuid):
        tag, idx, blk_off, bank = cache.extract_info(address)
        self.tag = tag
        self.idx = idx
        self.bank = bank
        self.misses.append(MSHRMiss(address, blk_off, is_write, data_in, uuid))


class Cache:
    def __init__(self, size, block_size, assoc, banks, mshr_k, dram = None):
        self.CS          = size       # Cache size in bits
        self.BLOCK_SIZE  = block_size # Block size in words
        self.A           = assoc      # Associativity
        self.NUM_BANKS   = banks      # Number of banks
        self.K           = mshr_k     # MSHR size
        self.NUM_TOTAL_WORDS = (self.CS // 8) // BYTES_PER_WORD
        self.NUM_TOTAL_FRAMES = self.NUM_TOTAL_WORDS // self.BLOCK_SIZE
        self.NUM_FRAMES_PER_BANK = self.NUM_TOTAL_FRAMES // self.NUM_BANKS
        self.NUM_SETS_PER_BANK = self.NUM_FRAMES_PER_BANK // self.A
        self.NUM_FRAMES_PER_SET = self.NUM_FRAMES_PER_BANK // self.NUM_SETS_PER_BANK
        # print("Banks:", self.NUM_BANKS)
        # print("-> Sets:", self.NUM_SETS_PER_BANK)
        # print("   -> Frames:", self.NUM_FRAMES_PER_SET)     
        # print("      -> Words:", self.BLOCK_SIZE)
        # print("         -> Bits:", WORD_SIZE)
        
        
        # self.FRAMES = CS//(BS * WORD_SIZE)
        # self.NUM_SETS = self.FRAMES // A
        self.INDEX_BITS = int(np.log2(self.NUM_SETS_PER_BANK))
        self.BLKOFF_BITS = int(np.log2(self.BLOCK_SIZE))
        self.BANK_BITS = int(np.log2(self.NUM_BANKS))
        self.TAG_BITS = ADDR_BITS - self.INDEX_BITS - self.BLKOFF_BITS - self.BANK_BITS - 2

        # print("Index bits     : ", self.INDEX_BITS)
        # print("Block off. bits: ", self.BLKOFF_BITS)
        # print("Bank idx. bits : ", self.BANK_BITS)
        # print("Tag bits       : ", self.TAG_BITS)
        self.cache = [[[CacheFrame(block_size=self.BLOCK_SIZE) for _ in range(self.A)] for _ in range(self.NUM_SETS_PER_BANK)] for _ in range(self.NUM_BANKS)]
        self.bank_latencies = self.NUM_BANKS * [0]
        self.bank_current_mshr = self.NUM_BANKS * [None]
        self.dram = dram
        self.mshr = []
        self.replays = []
        self.blocking = False
    

    def extract_info(self, address):
        address = tobits([address], bit_len=ADDR_BITS)
        bank    = frombits(address[2 : 2 + self.BANK_BITS])
        blk_off = frombits(address[2 + self.BANK_BITS : 2 + self.BANK_BITS + self.BLKOFF_BITS])
        idx     = frombits(address[2 + self.BANK_BITS + self.BLKOFF_BITS: ADDR_BITS - self.TAG_BITS])
        tag     = frombits(address[ADDR_BITS - self.TAG_BITS: ])
        return tag, idx, blk_off, bank

    def main_loop(self, tick):
        # Decrement bank timers for DRAM due to miss; if any are done, perform LRU eject and output replay
        for b_i in range(self.NUM_BANKS):
            mshr_done = self.bank_current_mshr[b_i]
            if self.bank_latencies[b_i]:
                self.bank_latencies[b_i] -= 1
            elif mshr_done is not None: # MSHR latency reached 0
                if mshr_done.address in self.dram.keys(): # If data can be found in DRAM
                    for word_idx in range(self.BLOCK_SIZE):
                        dram_word_addr  = mshr_done.bank << 2
                        dram_word_addr += word_idx       << (2 + self.BANK_BITS)
                        dram_word_addr += mshr_done.idx  << (2 + self.BANK_BITS + self.BLKOFF_BITS)
                        dram_word_addr += mshr_done.tag  << (ADDR_BITS - self.TAG_BITS)
                        data_from_dram = self.dram[dram_word_addr]
                        self.instant_write(dram_word_addr, data_from_dram, tick)
                else: # Else, eject old, and create fresh block
                    lru = 0
                    min_age = tick + 1
                    for i, frame in enumerate(self.cache[mshr_done.bank][mshr_done.idx]):
                        if frame.lru_tick < min_age : 
                            min_age = frame.lru_tick
                            lru = i
                    lru_frame = self.cache[mshr_done.bank][mshr_done.idx][lru]
                    for word_idx in range(self.BLOCK_SIZE):
                        dram_word_addr  = mshr_done.bank << 2
                        dram_word_addr += word_idx       << (2 + self.BANK_BITS)
                        dram_word_addr += mshr_done.idx  << (2 + self.BANK_BITS + self.BLKOFF_BITS)
                        dram_word_addr += lru_frame.tag  << (ADDR_BITS - self.TAG_BITS)
                        self.dram[dram_word_addr] = deepcopy(lru_frame.data[word_idx])
                    self.cache[mshr_done.bank][mshr_done.idx][lru] = CacheFrame(tick = tick)
                print(f"DRAM waiting finished with {len(mshr_done.misses)} misses")
                self.replays.extend(mshr_done.misses)
                self.bank_current_mshr[b_i] = None

        if len(self.mshr) == 0: return
        entry = self.mshr[0]
        
        # If bank for first MSHR element is free, pop element and begin DRAM retrieval timer
        if self.bank_current_mshr[entry.bank] is None:
            self.bank_latencies[entry.bank] = DRAM_LATENCY
            self.bank_current_mshr[entry.bank] = self.mshr.pop(0)

    def request_replay(self, uuid, tick):
        for i, to_replay in enumerate(self.replays):
            if uuid == to_replay.uuid:
                instr = self.replays.pop(i)
                return self.request(instr.address, instr.is_write, instr.data_in, tick)

    def request(self, address, is_write, data_in, tick):
        if self.blocking:
            if self.replays: return "Stall"
            for mshr in self.bank_current_mshr:
                if mshr: return "Stall"
        
        tag, idx, blk_off, bank = self.extract_info(address)
        for frame in self.cache[bank][idx]:
            if (not is_write and frame.valid and tag == frame.tag) or (is_write and ((not frame.valid and not frame.dirty) or (frame.valid and tag == frame.tag))):
                frame.lru = tick
                Cache.print_message("~ Hit      :", address, tag, idx, blk_off, bank)
                if is_write:
                    dcache.instant_write(address, data_in, tick)
                    return data_in
                else:
                    data_out = deepcopy(frame.data[blk_off])
                    return data_out
        

        # Check MSHR buffer entries, including those currently being 'serviced' by each bank
        for entry in self.bank_current_mshr:
            if entry is None: continue
            if (tag, idx, bank) == (entry.tag, entry.idx, entry.bank):
                entry.add_miss(address, self, is_write, data_in, tick)
                return "Miss"

        if self.mshr is not None:
            for entry in self.mshr:
                if (tag, idx, bank) == (entry.tag, entry.idx, entry.bank):
                    entry.add_miss(address, self, is_write, data_in, tick)
                    return "Miss"
        
        if len(self.mshr) == self.K:
            return "Stall"
        Cache.print_message("~ Miss     :", address, tag, idx, blk_off, bank)
        self.mshr.append(MSHREntry(address, self, is_write, data_in, uuid = tick))
        return "Miss"

    def instant_read(self, address, tick):
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
                self.instant_write(dram_word_addr, data_from_dram, tick)
            return data

    def instant_write(self, address, data, tick): 
        tag, idx, blk_off, bank = self.extract_info(address)
        cache_set = self.cache[bank][idx]
        lru = 0
        min_age = cache_set[lru].lru_tick
        for i, frame in enumerate(cache_set):
            if frame.valid and frame.tag == tag: # In place write
                frame.tag = tag
                frame.address = address - (blk_off << (2 + self.BANK_BITS))
                frame.data[blk_off] = data
                frame.lru_tick = tick
                frame.valid = True
                frame.dirty = True
                Cache.print_message("~ Write    :", address, tag, idx, blk_off, bank)
                # print("Data", frombits(data))
                return
            if frame.lru_tick < min_age : 
                min_age = frame.lru_tick
                lru = i

        lru_frame = cache_set[lru]
        if lru_frame.dirty:  # Write back
            Cache.print_message("~ Write LRU:", address, tag, idx, blk_off, bank)
            for word_idx in range(self.BLOCK_SIZE):
                dram_word_addr  = bank          << 2
                dram_word_addr += word_idx      << (2 + self.BANK_BITS)
                dram_word_addr += idx           << (2 + self.BANK_BITS + self.BLKOFF_BITS)
                dram_word_addr += lru_frame.tag << (ADDR_BITS - self.TAG_BITS)
                self.dram[dram_word_addr] = deepcopy(lru_frame.data[word_idx])
                Cache.print_message("    -> DRAM:", dram_word_addr, lru_frame.tag, idx, word_idx, bank)
        else:
            Cache.print_message("~ Write new:", address, tag, idx, blk_off, bank)
        
        lru_frame.tag = tag
        lru_frame.address = address - (blk_off << (2 + self.BANK_BITS))
        lru_frame.data[blk_off] = data
        lru_frame.lru_tick = tick
        lru_frame.valid = True
        lru_frame.dirty = True

    @staticmethod
    def print_message(message, address, tag, idx, blkoffset, bank):
        if __debug__:
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
    def instant_write_test(dcache, num_test, tick):
        print("Write test")
        for i in range(num_test):
            dcache.write(i * n, tobits([i], bit_len=WORD_SIZE), tick)
            tick += 1
        return tick

    def instant_read_test(dcache, num_test, tick):
        print("Read test")
        for i in range(num_test - 1, -1, -1):
            assert i == frombits(dcache.instant_read(i * n, tick)), f"Incorrect read at address {hex(i * n)}"
            tick += 1
        return tick

    def non_block_write_test(dcache, num_test, tick):
        print("Non-blocking Write test")
        i_in  = 0
        i_out = 0
        while i_out < num_test and tick < 5000:
            status_str = f"--- Tick {tick}   MSHR: {len(dcache.mshr)}/{dcache.K}"
            for b_i, bank in enumerate(dcache.cache):
                status_str += f"  B{b_i}: +{dcache.bank_latencies[b_i]}".ljust(4, ' ')
            print(status_str)
            
            if i_in < num_test: # Writes still left to request
                write_result = dcache.request(i_in * n, is_write = True, 
                                        data_in = tobits([i_in], bit_len=WORD_SIZE), tick = tick)
                if write_result != "Stall":
                    if write_result != "Miss":
                        read_result =  dcache.request(i_in * n, is_write = False, data_in = None, tick = tick)
                        assert frombits(read_result) == i_in, f"Incorrect read at address {hex(i_in * n)}: Expected {i_in}, {frombits(read_result)}"
                        i_out += 1

                    i_in += 1 # Write requested, but miss


            if len(dcache.replays):
                for miss in dcache.replays:
                    print("Replay:", hex(miss.address), "UUID:", miss.uuid)
                    request = dcache.request_replay(miss.uuid, tick = tick)
                    # print(request)
                    # print(dcache)
                    in_cache = frombits(request)
                    assert frombits(miss.data_in) == in_cache, f"Incorrect write at address {miss.address}: Expected {miss.address // n}, {in_cache}"
                    i_out += 1
            
            dcache.main_loop(tick)
            tick += 1
        return tick

    def blocking_write_test(dcache, num_test, tick):
        print("Blocking Write test")
        i_in  = 0
        i_out = 0
        while i_out < num_test and tick < 5000:
            status_str = f"--- Tick {tick}   MSHR: {len(dcache.mshr)}/{dcache.K}"
            for b_i, bank in enumerate(dcache.cache):
                status_str += f"  B{b_i}: +{dcache.bank_latencies[b_i]}".ljust(4, ' ')
            print(status_str)
            
            # Prioritize replaying of old miss, rather than taking new requests
            if len(dcache.replays):
                for miss in dcache.replays:
                    print("Replay:", hex(miss.address), "UUID:", miss.uuid)
                    request = dcache.request_replay(miss.uuid, tick)
                    assert request != "Stall", f"Write replay at address {hex(miss.address)} encountered stall"
                    in_cache = frombits(request)
                    assert (miss.address // n) == in_cache, f"Incorrect write at address {miss.address}: Expected {miss.address // n}, {in_cache}"
                    i_out += 1
            else:
                if i_in < num_test: # Writes still left to request
                    write_result = dcache.request(i_in * n, is_write = True, 
                                            data_in = tobits([i_in], bit_len=WORD_SIZE), tick = tick)
                    if write_result != "Stall":
                        if write_result != "Miss":
                            read_result =  dcache.request(i_in * n, is_write = False, data_in = None, tick = tick)
                            assert frombits(read_result) == i_in, f"Incorrect read at address {hex(i_in * n)}: Expected {i_in}, {frombits(read_result)}"
                            i_out += 1

                        i_in += 1 # Write requested, but miss


            dcache.main_loop(tick)
            tick += 1
        return tick
    
    def non_blocking_read_test(dcache, num_test, tick, tick_after_writing):
        print("Non-blocking Read test")
        i_in  = num_test - 1
        i_out = num_test - 1
        while i_out >= 0 and tick - tick_after_writing < 5000:
            status_str = f"--- Tick {tick}   MSHR: {len(dcache.mshr)}/{dcache.K}"
            for b_i, bank in enumerate(dcache.cache):
                status_str += f"  B{b_i}: +{dcache.bank_latencies[b_i]}".ljust(4, ' ')
            print(status_str)
            
            if i_in >= 0: # Reads still left to request
                result = dcache.request(i_in * n, is_write = False, data_in = None, tick = tick)
                if result != "Stall":
                    if result != "Miss":
                        assert frombits(result) == i_in # Read hit
                        i_out -= 1
                    i_in -= 1 # Read requested, but miss

            if len(dcache.replays):
                for miss in dcache.replays:
                    print("Replay:", hex(miss.address), "UUID:", miss.uuid)
                    request = dcache.request_replay(miss.uuid, tick = tick)
                    # print(request)
                    # print(dcache)
                    in_cache = frombits(request)
                    assert (miss.address // n) == in_cache, f"Incorrect read at address {miss.address}: Expected {frombits(miss.data_in)}, {in_cache}"
                    i_out -= 1

            dcache.main_loop(tick)
            tick += 1

    def blocking_read_test(dcache, num_test, tick, tick_after_writing):
        print("Non-blocking Read test")
        i_in  = num_test - 1
        i_out = num_test - 1
        while i_out >= 0 and tick - tick_after_writing < 5000:
            status_str = f"--- Tick {tick}   MSHR: {len(dcache.mshr)}/{dcache.K}"
            for b_i, bank in enumerate(dcache.cache):
                status_str += f"  B{b_i}: +{dcache.bank_latencies[b_i]}".ljust(4, ' ')
            print(status_str)
            
            if len(dcache.replays):
                for miss in dcache.replays:
                    print("Replay:", hex(miss.address), "UUID:", miss.uuid)
                    request = dcache.request_replay(miss.uuid, tick = tick)
                    assert request != "Stall", f"Read replay at address {hex(miss.address)} encountered stall"
                    # print(request)
                    # print(dcache)
                    in_cache = frombits(request)
                    assert (miss.address // n) == in_cache, f"Incorrect read at address {miss.address}: Expected {miss.address // n}, {in_cache}"
                    i_out -= 1
            else:
                if i_in >= 0: # Reads still left to request
                    result = dcache.request(i_in * n, is_write = False, data_in = None, tick = tick)
                    if result != "Stall":
                        if result != "Miss":
                            assert frombits(result) == i_in # Read hit
                            i_out -= 1
                        i_in -= 1 # Read requested, but miss

            dcache.main_loop(tick)
            tick += 1

    tick = 0
    dram = {}
    dcache = Cache(size=2048, block_size=4, assoc=2, banks=4, mshr_k=8, dram=dram)
    dcache.blocking = False

    # Addressing test
    for i in range(0, dcache.NUM_TOTAL_WORDS, 4):
        tag, idx, blk_off, bank = dcache.extract_info(i)
        print(f"Addr:  {hex(i)} Tag: {int(tag)} Idx: {int(idx)} Blk_off: {blk_off} Bank {int(bank)}")

    num_test = dcache.NUM_TOTAL_WORDS * 2

    test_block_offest = True
    n = 4 if test_block_offest else 8

    # tick = instant_write_test(dcache, num_test, tick)
    
    if not dcache.blocking: tick = non_block_write_test(dcache, num_test, tick)
    else:                   tick = blocking_write_test(dcache, num_test, tick)

    tick_after_writing = tick
    
    print("Cache after write test")
    print(dcache)

    print("DRAM after write test")
    for addr, data in sorted(dram.items()):
        print(hex(addr).ljust(5, ' '), frombits(data))


    # tick_after_reading = instant_read_test(dcache, num_test, tick)
    if not dcache.blocking: tick_after_reading = non_blocking_read_test(dcache, num_test, tick, tick_after_writing)
    else:                   tick_after_reading = blocking_read_test(dcache, num_test, tick, tick_after_writing)

    print("Cache after read test")
    print(dcache)

    print("DRAM after read test")
    for addr, data in sorted(dram.items()):
        print(hex(addr).ljust(5, ' '), frombits(data))
