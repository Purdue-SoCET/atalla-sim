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
        self.PE_total = (self.weight * self.input_from_left) + self.total_from_above
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
            for j in range(self.size):
                fifo.data[2 * self.size - i -j - 1] = input[i, self.size - j - 1]

    def load_partial_sum_FIFOs(self, PS):
        for i, fifo in enumerate(self.partial_sum_FIFOs):
            for j in range(self.size):
                fifo.data[2 * self.size - i -j - 1] = PS[i, self.size - j - 1]
        

    def read_output_FIFOs(self):
        out = np.zeros(shape = (self.size, self.size), dtype = self.dtype)
        for i, fifo in enumerate(self.output_FIFOs):
            out[:, i] = fifo.data[self.size - i : 2 * self.size - i]
        return out.T


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

# For correctness verification
def custom_matmul(A, B, dtype):
    size = A.shape[0]
    out = np.zeros(shape = (size, size), dtype = dtype)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                out[i, j] += (A[i, k] * B[k, j]).astype(dtype)
    return out


if __name__ == "__main__":
    N = 4
    dt = np.float16
    
    num_test = 5000

    sys = SystolicArray(size = N, dtype = dt)
    W =  np.random.rand(N, N).astype(dt)
    sys.load_weights(W)

    tick = 0
    initial_wait = (3 * N - 1) + 1
    num_read = 0
    num_sent = 0
    inputs   = [np.random.rand(N, N).astype(dt) for _ in range(num_test)]
    partials = [np.random.rand(N, N).astype(dt) for _ in range(num_test)]

    while (num_read < num_test):
        if tick % N == 0:
            if num_sent < num_test:
                # print(f"tick {tick}: loading input, ps")
                sys.load_input_FIFOs(inputs[num_sent])
                sys.load_partial_sum_FIFOs(partials[num_sent])
                num_sent += 1
        
        if (tick - initial_wait >= 0) and ((tick - initial_wait) % N == 0):
            out = sys.read_output_FIFOs()
            expected = custom_matmul(W, inputs[num_read], dt) + partials[num_read]
            num_read += 1
            
            assert np.allclose(out, expected, rtol = 1e-2)
        
        # print("~~~INPUT FIFO")
        # for f in sys.input_FIFOs:
        #     s = ""
        #     for n in f.data: s += str(n).ljust(8, " ")
        #     print(s)

        sys.cycle()
        tick += 1

