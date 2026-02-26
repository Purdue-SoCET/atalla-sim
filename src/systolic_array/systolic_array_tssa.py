from typing import List, Optional

from base.clocked_object import Clocked
from base.queue import SimQueue

class PE:
    """Single processing element in the systolic mesh.

    links: [0: left, 1: up, 2: right, 3: down]
    """

    def __init__(self, links: Optional[List[Optional["PE"]]] = None):
        if links is None:
            links = [None, None, None, None]
        self.links: List[Optional["PE"]] = links
        self.activation_latch: float = 0.0
        self.weight: float = 0.0
        self.accumulation: float = 0.0
        self.mul_reg: float = 0.0

    def _input(self, activation: float) -> None:
        self.activation_latch = float(activation)

    def _weight(self, weight: float) -> None:
        self.weight = float(weight)

    def _accumulation(self, accumulation: float) -> None:
        self.accumulation = float(accumulation)

    def shift(self, shift_direction: int, stream: str = "activation") -> None:
        neighbor = self.links[shift_direction]
        if neighbor is None:
            return

        if stream == "activation":
            neighbor._input(self.activation_latch)
            return
        if stream == "weight":
            neighbor._weight(self.weight)
            return
        if stream == "accumulation":
            neighbor._accumulation(self.accumulation)
            return
        raise ValueError("stream must be one of: activation, weight, accumulation")


class SystolicArrayTSSA(Clocked):
    def __init__(self, size: int, boundary_buffer_depth: int = 32, dtype: Optional[object] = None):
        super().__init__()
        self.size = int(size)
        self.dtype = dtype
        self.psum_output_fifo_bottom: List[List[float]] = []
        self.array: List[List[PE]] = self._setup_array()
        self._input_fifo_left: List[SimQueue[float]] = [SimQueue(boundary_buffer_depth) for _ in range(self.size)]
        self._weight_boundary: List[SimQueue[float]] = [SimQueue(boundary_buffer_depth) for _ in range(self.size)]
        self._psum_input_fifo_top: List[SimQueue[float]] = [SimQueue(boundary_buffer_depth) for _ in range(self.size)]

        # Top-level controls (driven by GSAU/controller in the full design).
        self.weight_en: bool = False
        self.mac_shift: bool = True
        self.start: bool = False
        self.stall: bool = False
        self.value_ready: bool = False

        # Two-cycle control latency tracking for MAC valid.
        self._start_pipe_0: bool = False
        self._start_pipe_1: bool = False

    def enqueue(self, activations: List[float]) -> bool:
        """Enqueue one activation vector into input FIFO (left boundary), one value per row."""
        if len(activations) != self.size:
            raise ValueError("activations must match systolic array size")

        if any(q.is_full() for q in self._input_fifo_left):
            return False

        for i in range(self.size):
            self._input_fifo_left[i].enqueue(float(activations[i]))
        return True

    def enqueue_weights(self, weights: List[float]) -> bool:
        """Enqueue one weight vector to preload row-stationary weights through shared pass buses."""
        if len(weights) != self.size:
            raise ValueError("weights must match systolic array size")
        if any(q.is_full() for q in self._weight_boundary):
            return False
        for i in range(self.size):
            self._weight_boundary[i].enqueue(float(weights[i]))
        return True

    def enqueue_psums(self, psums: List[float]) -> bool:
        """Optional psum input FIFO injection (top boundary), one value per column."""
        if len(psums) != self.size:
            raise ValueError("psums must match systolic array size")
        if any(q.is_full() for q in self._psum_input_fifo_top):
            return False
        for j in range(self.size):
            self._psum_input_fifo_top[j].enqueue(float(psums[j]))
        return True

    def set_control(
        self,
        *,
        weight_en: Optional[bool] = None,
        mac_shift: Optional[bool] = None,
        start: Optional[bool] = None,
        stall: Optional[bool] = None,
    ) -> None:
        if weight_en is not None:
            self.weight_en = bool(weight_en)
        if mac_shift is not None:
            self.mac_shift = bool(mac_shift)
        if start is not None:
            self.start = bool(start)
        if stall is not None:
            self.stall = bool(stall)

    def _setup_array(self) -> List[List[PE]]:
        array = [[PE() for _ in range(self.size)] for _ in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                left = array[i][j - 1] if j > 0 else None
                up = array[i - 1][j] if i > 0 else None
                right = array[i][j + 1] if j < self.size - 1 else None
                down = array[i + 1][j] if i < self.size - 1 else None
                array[i][j].links = [left, up, right, down]
        return array

    def load_weights(self, weights: List[List[float]]) -> None:
        if len(weights) != self.size or any(len(row) != self.size for row in weights):
            raise ValueError("weights must be size x size")
        for i in range(self.size):
            for j in range(self.size):
                self.array[i][j].weight = float(weights[i][j])

    def tick(self, time: Optional[float] = None) -> None:
        # Stall freezes forward progress so values/ready can be preserved for lossless backpressure.
        if self.stall:
            return

        old_act = [[self.array[i][j].activation_latch for j in range(self.size)] for i in range(self.size)]
        old_w = [[self.array[i][j].weight for j in range(self.size)] for i in range(self.size)]
        old_acc = [[self.array[i][j].accumulation for j in range(self.size)] for i in range(self.size)]
        old_mul = [[self.array[i][j].mul_reg for j in range(self.size)] for i in range(self.size)]

        # Shared rightward pass bus is muxed between weight preload and activation shift.
        if self.weight_en:
            for i in range(self.size):
                in_val = self._weight_boundary[i].dequeue()
                pass_bus = float(in_val) if in_val is not None else 0.0
                for j in range(self.size):
                    prev_weight = old_w[i][j]
                    self.array[i][j].weight = pass_bus
                    pass_bus = prev_weight
        elif self.mac_shift:
            for i in range(self.size):
                in_val = self._input_fifo_left[i].dequeue()
                pass_bus = float(in_val) if in_val is not None else 0.0
                for j in range(self.size):
                    prev_act = old_act[i][j]
                    self.array[i][j].activation_latch = pass_bus
                    pass_bus = prev_act

        # Stage 2: add registered product with vertical psum input.
        for i in range(self.size):
            for j in range(self.size):
                top_boundary = self._psum_input_fifo_top[j].dequeue() if i == 0 else None
                psum_in = float(top_boundary) if top_boundary is not None else (0.0 if i == 0 else old_acc[i - 1][j])
                self.array[i][j].accumulation = float(old_mul[i][j] + psum_in)

        # Stage 1: combinational multiply, product captured into register for next cycle.
        for i in range(self.size):
            for j in range(self.size):
                if self.start:
                    self.array[i][j].mul_reg = float(
                        self.array[i][j].activation_latch * self.array[i][j].weight
                    )

        # value_ready is asserted two cycles after start.
        self.value_ready = self._start_pipe_1
        self._start_pipe_1 = self._start_pipe_0
        self._start_pipe_0 = self.start

        if self.value_ready:
            bottom_row = [self.array[self.size - 1][j].accumulation for j in range(self.size)]
            self.psum_output_fifo_bottom.append(bottom_row)

    def get_buffer(self) -> List[List[float]]:
        return self.psum_output_fifo_bottom

    def boundary_levels(self) -> List[int]:
        """Current fill level of each input FIFO (left boundary)."""
        return [len(q) for q in self._input_fifo_left]
