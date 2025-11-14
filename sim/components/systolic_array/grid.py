import numpy as np
from typing import Tuple

def conv_output_shape(H_in:int, W_in:int, Kh:int, Kw:int, stride:int=1, pad:int=0, dilation:int=1) -> Tuple[int,int]:
    H_out = (H_in + 2*pad - dilation*(Kh-1) - 1)//stride + 1
    W_out = (W_in + 2*pad - dilation*(Kw-1) - 1)//stride + 1
    if H_out <= 0 or W_out <= 0:
        raise ValueError(f"Non-positive output shape computed: H_out={H_out}, W_out={W_out}. Check params.")
    return H_out, W_out


class PE:
    def __init__(self, links: list | None = None) -> None:
        if links is None:
            links = [None, None, None, None]
        self.links = links
        self.activation: float = 0.0
        self.weight: float = 0.0
        self.partial_sum: float = 0.0
        self.weight_str: str = '--'
        self.activation_str: str = '0'
        self.accumulation_str: str = '0'

    def load_weight(self, weight: float) -> None:
        """Load a numeric weight into this PE and update the debug string."""
        self.weight = weight
        self.weight_str = f"{weight}"

    def set_activation(self, activation: float) -> None:
        """Set the current activation for this PE and update the debug string."""
        self.activation = activation
        self.activation_str = f"{activation}"

    def cycle(self, accum_in: float) -> float:
        psum = accum_in + self.weight * self.activation
        self.partial_sum = psum
        mul_term = f"{self.activation_str}*{self.weight_str}" if self.activation_str not in ['0', '0.0'] and self.weight_str != '--' else '0'
        if accum_in == 0.0:
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

    def _reset_state(self) -> None:
        for i in range(self.size):
            for j in range(self.size):
                pe = self.array[i][j]
                # Reset numeric and debug states
                pe.activation = 0.0
                pe.activation_str = '0'
                pe.partial_sum = 0.0
                pe.accumulation_str = '0'
        # Clear input queues and output buffer
        self.input_fifo = [[] for _ in range(self.size)]
        self.output_buffer = []

    def clear(self) -> None:
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
        for i in range(self.size):
            for j in range(self.size):
                self.array[i][j].load_weight(0.0)
                self.weights[i][j] = 0.0
        for k in range(K_t):
            for m in range(M_t):
                w_val = float(Wt[k, m])
                self.array[k][m].load_weight(w_val)
                self.weights[k][m] = w_val

    def feed_inputs(self, inputs: list[float]) -> None:
        assert len(inputs) == self.size, "Input vector length must match array size"
        for row, val in enumerate(inputs):
            self.input_fifo[row].append(float(val))

    def cycle(self) -> list[float]:
        size = self.size
        new_acts: list[list[float]] = [[0.0 for _ in range(size)] for _ in range(size)]
        for i in range(size):
            for j in range(size-1, 0, -1):
                new_acts[i][j] = self.array[i][j-1].activation
            if self.input_fifo[i]:
                new_acts[i][0] = self.input_fifo[i].pop(0)
            else:
                new_acts[i][0] = 0.0
        new_psums: list[list[float]] = [[0.0 for _ in range(size)] for _ in range(size)]
        for i in range(size):
            for j in range(size):
                self.array[i][j].set_activation(new_acts[i][j])
            for j in range(size):
                if i == 0:
                    accum_in = 0.0
                else:
                    accum_in = new_psums[i-1][j]
                psum = self.array[i][j].cycle(accum_in)
                new_psums[i][j] = psum
        bottom_row = new_psums[size - 1][:]
        self.output_buffer.append(bottom_row)
        return bottom_row

