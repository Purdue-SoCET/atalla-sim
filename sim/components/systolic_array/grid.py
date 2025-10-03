import numpy as np
from typing import Tuple

class PE:
    def __init__(self, links = None):
        # Neighboring PEs: 0=left, 1=up, 2=right, 3=down
        if links is None: links = [None, None, None, None]
        self.links = links

        self.activation: float = 0.0
        self.weight: float = 0.0
        self.partial_sum: float = 0.0

        self.weight_str: str = '--'
        self.activation_str: str = '0'
        self.accumulation_str: str = '0'

    def reset(): 
        self.activation: float = 0.0
        self.weight: float = 0.0
        self.partial_sum: float = 0.0

    def load_weight(self, weight: float) -> None:
        self.weight = weight
        self.weight_str = f"{weight}"

    def set_activation(self, activation: float) -> None:
        self.activation = activation
        self.activation_str = f"{activation}"

    def _input(self, accum_in: float, activation: float, activation_vld: bool, weight: float, weight_vld: bool): 


    def _output(self, ): 

    def cycle(self, accum_in: float) -> float:
        psum = accum_in + self.weight * self.activation
        # Update internal numeric state
        self.partial_sum = psum
        # Update debug accumulation string
        mul_term = f"{self.activation_str}*{self.weight_str}" if self.activation_str not in ['0', '0.0'] and self.weight_str != '--' else '0'
        if accum_in == 0.0:
            # no previous accumulation
            self.accumulation_str = mul_term
        else:
            prev_str = self.accumulation_str
            if prev_str.strip() == '0' and mul_term == '0':
                self.accumulation_str = '0'
            elif prev_str.strip() == '0':
                self.accumulation_str = mul_term
            elif mul_term == '0':
                self.accumulation_str = prev_str
            else:
                self.accumulation_str = prev_str + ' + ' + mul_term
        return psum


class SystolicArray:
    def __init__(self, size: int) -> None:
        self.size = size
        self.input_fifo: list[list[float]] = [[] for _ in range(size)]
        self.output_buffer: list[list[float]] = []
        self.array = self._setup_array()
        self.weights = [[0.0 for _ in range(size)] for _ in range(size)]

    def _setup_array(self) -> list[list[PE]]:
        array: list[list[PE]] = [[PE() for _ in range(self.size)] for _ in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                left = array[i][j-1] if j > 0 else None
                up = array[i-1][j] if i > 0 else None
                right = array[i][j+1] if j < self.size - 1 else None
                down = array[i+1][j] if i < self.size - 1 else None
                array[i][j].links = [left, up, right, down]
        return array

    def reset(self):
        for i in range(self.size):
            for j in range(self.size):
                self.array[i][j].reset()
        self.input_fifo = [[] for _ in range(self.size)]
        self.output_buffer = []

    def clear(self):
        for i in range(self.size):
            for j in range(self.size):
                pe = self.array[i][j]
                # Reset numeric and debug states
                pe.activation = 0.0
                pe.activation_str = '0'
                pe.weight = 0.0
                pe.weight_str = '--'
                pe.partial_sum = 0.0
                pe.accumulation_str = '0'
                self.weights[i][j] = 0.0
        # Clear queues and output buffer
        self.input_fifo = [[] for _ in range(self.size)]
        self.output_buffer = []

    def preload_weights(self, Wt: np.ndarray) -> None:
        K_t, M_t = Wt.shape
        assert K_t <= self.size and M_t <= self.size, "Weight tile exceeds array size"
        # Clear previous weights and load numeric values into PE weights and into self.weights
        for i in range(self.size):
            for j in range(self.size):
                # Reset the PE's weight and debug fields
                self.array[i][j].load_weight(0.0)
                self.weights[i][j] = 0.0
        for k in range(K_t):
            for m in range(M_t):
                # Store numeric weight in both the PE and the weight matrix
                w_val = float(Wt[k, m])
                self.array[k][m].load_weight(w_val)
                self.weights[k][m] = w_val

    def feed_inputs(self, inputs: list[float]) -> None:
        assert len(inputs) == self.size, "Input vector length must match array size"
        for row, val in enumerate(inputs):
            self.input_fifo[row].append(float(val))

    def cycle(self):
        size = self.size
        # Step 1: determine next activations for each PE by shifting and inserting new inputs
        # We'll store the new activations in a temporary matrix
        new_acts: list[list[float]] = [[0.0 for _ in range(size)] for _ in range(size)]
        for i in range(size):
            # Shift activations to the right across the row
            for j in range(size-1, 0, -1):
                new_acts[i][j] = self.array[i][j-1].activation
            # Insert new input from the FIFO at column 0
            if self.input_fifo[i]:
                new_acts[i][0] = self.input_fifo[i].pop(0)
            else:
                new_acts[i][0] = 0.0
        # Step 2: perform MAC operation on each PE row by row
        # We'll compute new partial sums and update the PE's activation and partial_sum fields
        # We also maintain a matrix of new partial sums to determine the accumulation input for the next row
        new_psums: list[list[float]] = [[0.0 for _ in range(size)] for _ in range(size)]
        for i in range(size):
            for j in range(size):
                # Update the PE's activation with the newly shifted value
                self.array[i][j].set_activation(new_acts[i][j])
            # Now compute partial sums across columns for this row
            for j in range(size):
                # The accumulation input is the partial sum from the PE above in the same column
                if i == 0:
                    accum_in = 0.0
                else:
                    accum_in = new_psums[i-1][j]
                psum = self.array[i][j].cycle(accum_in)
                new_psums[i][j] = psum
        # Step 3: save bottom row partial sums to the output buffer
        bottom_row = new_psums[size - 1][:]
        self.output_buffer.append(bottom_row)
        # Step 4: return the bottom row for convenience
        # The PE partial sums have been updated internally for the next cycle through ``cycle`` calls
        return bottom_row

    def vertical_psum(self, Wt: np.ndarray, b_vec: np.ndarray) -> np.ndarray:
        """
        **Deprecated.**

        This method previously performed a direct dot-product across the contraction
        dimension (k) to simulate a vertical accumulation without modelling the
        systolic dataflow. It has been superseded by the cycle-based simulation
        implemented in ``SystolicArray.cycle``. The new simulation preloads
        weights into the array and then feeds activations into the leftmost
        column on each cycle. Activations shift to the right while partial
        sums accumulate vertically and propagate downward. Results are emitted
        from the bottom row.

        The current implementation raises a ``NotImplementedError`` to warn
        against inadvertent use. Please use the cycle-based methods instead.
        """
        raise NotImplementedError(
            "vertical_psum() is deprecated; use SystolicArray.cycle() with preloaded weights "
            "and feed_inputs() to stream activations for accurate systolic simulation."
        )