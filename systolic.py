import numpy as np

class FIFO():
    def __init__(self, size, dtype : np.dtype):
        self.size = size
        self.dtype = dtype
        self.data = [dtype(0) for _ in range(size)]
    
    def read(self):
        temp = self.data[-1]
        self.write(self.dtype(0))
        return temp
    
    def write(self, num):
        self.data.insert(0, self.dtype(num))
        self.data = self.data[:-1]

class MAC():
    def __init__(self, dtype : np.dtype, latency : int):
        self.weight = dtype(0)
        self.PE_total = dtype(0)
        self.latency = latency
        self.input_from_left = dtype(0)
        self.total_from_above = dtype(0)
        self.output_right = dtype(0)
        self.output_down = dtype(0)

    def compute(self):
        self.PE_total = self.weight * self.input_from_left + self.total_from_above
        self.output_down = self.PE_total
        self.output_right = self.input_from_left
    
class SystolicArray():
    def __init__(self, size : int, dtype : np.dtype):
        self.size = size
        self.dtype = dtype
        self.MACs = [[MAC(dtype, 3) for _ in range(size)] for _ in range(size)]
        self.partial_sum_FIFOs = [FIFO(size = 2 * size, dtype = dtype) for _ in range(size)]
        self.input_FIFOs =       [FIFO(size = 2 * size, dtype = dtype) for _ in range(size)]
        self.output_FIFOs =      [FIFO(size = 2 * size, dtype = dtype) for _ in range(size)]

    def load_weights(self, weight):
        weight = weight.T
        for i, row in enumerate(weight):
            for j, num in enumerate(row):
                self.MACs[i][j].weight = num
    
    def load_input_FIFOs(self, input):
        for i, fifo in enumerate(self.input_FIFOs):
            fifo.data[2 * self.size - i : self.size - i] = input[i, ::-1]

    def load_partial_sum_FIFOs(self, PS):
        PS = PS.T
        for i, fifo in enumerate(self.partial_sum_FIFOs):
            fifo.data[2 * self.size - i: self.size - i] = PS[:, i]
        

    def read_output_FIFOs(self):
        out = np.zeros(shape = (self.size, self.size), dtype = self.dtype)
        for i, fifo in enumerate(self.output_FIFOs):
            out[:, i] = fifo.data[self.size - i : 2 * self.size - i]
        return out.T[:, ::-1]


    def cycle(self):
        # Write to output FIFOs
        for j in range(self.size):
            self.output_FIFOs[j].write(self.MACs[-1][j].output_down)

        # Read partial sum into top row
        for j in range(self.size):
            self.MACs[0][j].total_from_above = self.partial_sum_FIFOs[j].read()

        # Move outputs down
        for i in range(1, self.size):
            for j in range(self.size):
                self.MACs[i][j].total_from_above = self.MACs[i-1][j].output_down

        # Move inputs left
        for i in range(self.size):
            for j in range(1, self.size):
                self.MACs[i][j].input_from_left = self.MACs[i][j-1].output_right

        # Write new input into left column
        for i in range(self.size):
            self.MACs[i][0].input_from_left = self.input_FIFOs[i].read()
        
        #Perform MAC computations
        for i in range(self.size):
            for j in range(self.size):
                self.MACs[i][j].compute()


               

if __name__ == "__main__":
    N = 4
    dt = np.float16
    sys = SystolicArray(size = N, dtype = dt)
    W = np.array([[3.1415926, -0.5, 55.555, 12.345], 
                         [2.3, 4.555, -6.734, -0.94], 
                         [-00.0089, -3.0002, 21.11, 4.44], 
                         [-0.21718, -3.21, 1.212, 9]],
                        dtype = dt)
    I = np.array([[1, 5, 9, 3], 
                   [2, 6, 0, 4], 
                   [3, 7, 1, 5], 
                   [4, 8, 2, 6]], dtype = dt)
    PS = np.array([[1, 5, 9, 3], 
                   [2, 6, 0, 4], 
                   [3, 7, 1, 5], 
                   [4, 8, 2, 6]], dtype = dt)
    
    # W = np.zeros(shape = (N, N), dtype = dt)
    # I = np.zeros(shape = (N, N), dtype = dt)
    PS = np.zeros(shape = (N, N), dtype = dt)

    sys.load_weights(W)
    sys.load_input_FIFOs(I)
    sys.load_partial_sum_FIFOs(PS)

    for fifo in sys.input_FIFOs:
        print(*[str(n) for n in fifo.data], sep = " ")

    # The number of ticks it should take for 1 standalone GEMM
    num_ticks = sys.size * 3 - 1

    for tick in range(num_ticks + 1):
        print(tick)
        for i in range(N):
            s = ""
            for j in range(N):
                s += str(sys.MACs[i][j].input_from_left).ljust(5, " ")
            print(s)
        print()
        sys.cycle()
    
    out = sys.read_output_FIFOs()
    expected = np.matmul(W, I) + PS
    # print(out)
    if not np.allclose(out, expected):
        print("Out")
        print(out)
        print("Expected")
        print(expected)
    else:
        print("correct")
