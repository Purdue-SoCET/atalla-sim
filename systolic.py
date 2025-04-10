import numpy as np
# import matplotlib.pyplot as plt
    
class SystolicArray():
    def __init__(self, size : int, PE_latency : int, dtype : np.dtype):
        self.size = size
        self.PE_latency = PE_latency
        self.dtype = dtype
        self.weight            = np.zeros(shape=(size, size), dtype=dtype)
        self.PE_total          = np.zeros(shape=(size, size), dtype=dtype)
        self.input_from_left   = np.zeros(shape=(size, size), dtype=dtype)
        self.total_from_above  = np.zeros(shape=(size, size), dtype=dtype)
        self.output_right      = np.zeros(shape=(size, size), dtype=dtype)
        self.output_down       = np.zeros(shape=(size, size), dtype=dtype)
        self.partial_sum_FIFOs = np.zeros(shape=(2 * size, size), dtype=dtype)
        self.input_FIFOs       = np.zeros(shape=(size, 2 * size), dtype=dtype)
        self.output_FIFOs      = np.zeros(shape=(2 * size, size), dtype=dtype)

    def load_weights(self, weight):
        self.weight = weight.T
    
    def load_input_FIFOs(self, input):
        for i in range(self.size):
            self.input_FIFOs[i, self.size - i: 2 * self.size - i] = input[i, :]

    def load_partial_sum_FIFOs(self, PS):
        for i in range(self.size):
            self.partial_sum_FIFOs[self.size - i : 2 * self.size - i, i] = PS[i, :]
        
    def read_output_FIFOs(self):
        out = np.zeros(shape = (self.size, self.size), dtype = self.dtype)
        for i in range(self.size):
            out[i, :] = self.output_FIFOs[self.size - i : 2 * self.size - i, i]
        return out


    def cycle(self):
        # Write to output FIFOs
        self.output_FIFOs = np.roll(self.output_FIFOs, 1, axis = 0)
        self.output_FIFOs[0, :] = self.output_down[-1, :]

        # Move outputs down
        self.total_from_above = np.roll(self.output_down, 1, axis = 0)

        # Read partial sum into top row
        self.total_from_above[0,:] = self.partial_sum_FIFOs[-1, :]
        self.partial_sum_FIFOs = np.roll(self.partial_sum_FIFOs, 1, axis = 0)
        self.partial_sum_FIFOs[0, :] = self.dtype(0)

        # Move inputs right
        self.input_from_left = np.roll(self.output_right, 1, axis = 1)

        # Write new input into left column
        self.input_from_left[:, 0] = self.input_FIFOs[:, -1]
        self.input_FIFOs = np.roll(self.input_FIFOs, 1, axis = 1)
        self.input_FIFOs[:, 0] = self.dtype(0)
        
        #Perform MAC computations
        self.PE_total = (self.weight * self.input_from_left).astype(self.dtype) + self.total_from_above
        self.output_down = self.PE_total
        self.output_right = self.input_from_left


if __name__ == "__main__":
    def weight_stationary_test(N, num_test, PE_latency):
        dt = np.float16

        sys = SystolicArray(size = N, PE_latency= PE_latency, dtype = dt)
        W =  np.random.rand(N, N).astype(dt) - 0.5
        sys.load_weights(W)

        tick = 0
        initial_wait = (3 * N - 1) + 1
        num_read = 0
        num_sent = 0
        inputs   = [np.random.rand(N, N).astype(dt) - 0.5 for _ in range(num_test)]
        partials = [np.random.rand(N, N).astype(dt) - 0.5 for _ in range(num_test)]

        while (num_read < num_test):
            if tick % N == 0:
                if num_sent < num_test:
                    sys.load_input_FIFOs(inputs[num_sent])
                    sys.load_partial_sum_FIFOs(partials[num_sent])
                    num_sent += 1
            
            if (tick - initial_wait >= 0) and ((tick - initial_wait) % N == 0):
                out = sys.read_output_FIFOs()
                expected = np.matmul(W, inputs[num_read]).astype(dt) + partials[num_read]
                num_read += 1
                print(f"Num read: {num_read}")
                if not np.allclose(out, expected, atol = N * 1e-3):
                    print(out)
                    print(expected)
                    assert False

            sys.cycle()
            tick += 1

    def tiled_matmul_test(Big_N, N, PE_latency, dt):
        assert Big_N >= N
        Big_W = np.random.rand(Big_N, Big_N).astype(dt) - 0.5
        Big_I = np.random.rand(Big_N, Big_N).astype(dt) - 0.5
        partials = np.zeros(shape = (Big_N, Big_N), dtype = dt)

        tiles = int(np.ceil(Big_N / N))
        sys = SystolicArray(size = N, PE_latency = PE_latency, dtype = dt)

        tick = 0
        initial_wait = (3 * N - 1) + 1
        num_read = 0
        num_sent = 0

        # print("Big W")
        # print(Big_W)
        # print("Big I")
        # print(Big_I)

        weight_row = 0
        while weight_row < tiles:
            term_idx = 0
            while term_idx < tiles:
                W = Big_W[N * weight_row : N * (weight_row+1), N * term_idx : N * (term_idx + 1)]
                sys.load_weights(W)
                tick += N
                for _ in range(N): sys.cycle()
                # print("New W")
                # print(W)
                input_col = 0
                partial_out_col = 0
                this_mul_wait = 0
                while partial_out_col < tiles:
                    if (this_mul_wait - initial_wait >= 0) and ((this_mul_wait - initial_wait) % N == 0):
                        out = sys.read_output_FIFOs()
                        # print("Partial: ")
                        # print(out)
                        partials[N * weight_row : N * (weight_row+1), N * partial_out_col : N * (partial_out_col+1)] += out
                        partial_out_col += 1
                    if (this_mul_wait % N == 0) and input_col < tiles:
                        I = Big_I[N * term_idx : N * (term_idx+1), N * input_col : N * (input_col+1)]
                        sys.load_input_FIFOs(I)
                        # print("New I")
                        # print(I)
                        input_col += 1
                    
                        #  += out
                    sys.cycle()
                    this_mul_wait += 1
                    tick += sys.PE_latency
                term_idx += 1
            weight_row += 1
            print(f"Output row {weight_row}/{tiles} done")
        # expected = custom_matmul(Big_W, Big_I, dt)
        expected = np.matmul(Big_W, Big_I).astype(dt)
        if not np.allclose(partials, expected, atol = 1e-3 * Big_N):
            diff = partials-expected
            print(diff)
            max_diff = np.max(np.abs(diff))
            print(f"Max diff: {max_diff}")
            indices = np.where(diff == max_diff)
            print(f"{partials[indices]} - {expected[indices]}")
            assert False

        return(tick)

    weight_stationary_test(128, 100, 18)

    dt = np.float16
    latency = 18
    Big_N = 128
    dims = [4, 8, 16, 32, 64, 128]
    results = {}
    for N in dims:
        results[N] = tiled_matmul_test(Big_N, N, latency, dt)

    print(results)
    # plt.plot(results.keys(), results.values(), 'o-')
    # plt.title(f"Tiled GEMM for {Big_N}x{Big_N} Matrix")
    # plt.xlabel("Systolic Array Dimension")
    # plt.ylabel("Compute Time")
    # plt.xscale('log', base = 2)
    # plt.yscale('log', base = 2)
    # # plt.savefig('results.png')
    # plt.grid()
    # plt.show()
    

