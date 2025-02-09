import numpy as np

def tobits(data, bit_len = 8): 
    bits = []
    for b in data:
        for i in range(bit_len - 1, -1, -1):
            bits.append((b >> i) & 1)
    bits.reverse()
    return bits

def frombits(data, signed = False):
    if data[-1] and signed: 
        return -1 * np.int32(frombits([int(not n) for n in data])) - 1
    else: 
        s = sum(b << i for i,b in enumerate(data))
        if signed: return np.int32(s) 
        else: return np.uint32(s)
   