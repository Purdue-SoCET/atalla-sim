import numpy as np

def tobits(data, bit_len = 8): 
    bits = []
    for b in data:
        for i in range(bit_len - 1, -1, -1):
            bits.append((b >> i) & 1)
    bits.reverse()
    return bits

def frombits(data, signed = False):
    if type(data) == int: return data
    if data == None: raise TypeError("Data cannot be NoneType")
    if type(data) == str: raise TypeError(f'Data must be a list of int, not a str "{data}"')
    if data == []: return 0
    if data[-1] and signed: 
        return -1 * np.int32(frombits([int(not n) for n in data])) - 1
    else: 
        s = sum(b << i for i,b in enumerate(data))
        if signed: return np.int32(s) 
        else: return np.uint32(s)

def load_instructions(filename):
    word_list = []
    with open(filename, "rb") as f:
        data = f.read()
    if len(data) % 4 != 0:
        raise ValueError("File length is not a multiple of 4 bytes.")
    for i in range(0, len(data), 4):
        word = data[i:i+4]
        word_list.append(word[::-1])
    return word_list