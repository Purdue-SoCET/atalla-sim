from typing import List, Optional
import math

from base.dtype import DType, cast_scalar, normalize_dtype

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
        self.mul_ops: int = 0
        self.add_ops: int = 0
        self.psum_adds: int = 0
        self.mac_ops: int = 0

    def _input(self, activation: float) -> None:
        self.activation_latch = float(activation)

    def _weight(self, weight: float) -> None:
        self.weight = float(weight)

    def _accumulation(self, accumulation: float) -> None:
        self.accumulation = float(accumulation)

    def count_mul(self) -> None:
        self.mul_ops += 1
        self.mac_ops += 1

    def count_add(self, is_psum: bool = False) -> None:
        self.add_ops += 1
        if is_psum:
            self.psum_adds += 1

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
        self.dtype = normalize_dtype(dtype, default=None)
        self._current_dtype: Optional[DType] = None
        self.algo_counts = {
            "act_rows": 0,
            "wgt_cols": 0,
            "out_rows": 0,
        }
        self.metrics = {
            "mul_ops": 0,
            "add_ops": 0,
            "psum_adds": 0,
            "mac_ops": 0,
        }
        self.internal_bytes = {
            "weight_shift": 0,
            "act_shift": 0,
            "psum_shift": 0,
            "output": 0,
        }
        self.internal_bytes_valid = {
            "weight_shift": 0,
            "act_shift": 0,
            "psum_shift": 0,
            "output": 0,
        }
        self.valid_mac_cycles: int = 0
        self.saturation_count: int = 0
        self.overflow_count: int = 0
        self.psum_output_fifo_bottom: List[List[float]] = []
        self.array: List[List[PE]] = self._setup_array()
        self._input_fifo_left: List[SimQueue[float]] = [SimQueue(boundary_buffer_depth) for _ in range(self.size)]
        self._weight_boundary: List[SimQueue[float]] = [SimQueue(boundary_buffer_depth) for _ in range(self.size)]
        self._psum_input_fifo_top: List[SimQueue[float]] = [SimQueue(boundary_buffer_depth) for _ in range(self.size)]
        self._algo_out_pending: int = 0

        # Top-level controls (driven by GSAU/controller in the full design).
        self.weight_en: bool = False
        self.mac_shift: bool = True
        self.start: bool = False
        self.stall: bool = False
        self.value_ready: bool = False

        # Two-cycle control latency tracking for MAC valid.
        self._start_pipe_0: bool = False
        self._start_pipe_1: bool = False

    def _resolve_dtype(self, dtype: Optional[object]) -> DType:
        dtype_norm = normalize_dtype(dtype, default=self.dtype)
        if dtype_norm is None:
            raise ValueError("dtype must be specified")
        if self._current_dtype is None:
            self._current_dtype = dtype_norm
        if dtype_norm != self._current_dtype:
            raise ValueError("tssa dtype mismatch: job=%s req=%s" % (self._current_dtype, dtype_norm))
        return dtype_norm

    def _dtype_bytes(self) -> int:
        dtype = self._current_dtype or self.dtype
        if dtype == DType.INT8:
            return 1
        return 2

    def _note_cast(self, value: float, cast_value: float) -> None:
        if self._current_dtype == DType.FP16:
            if abs(float(value)) > 65504.0:
                self.saturation_count += 1
            if not math.isfinite(float(cast_value)):
                self.overflow_count += 1

    def algo_flops(self) -> int:
        # Count algorithmic GEMM flops from observed streamed dimensions.
        # M = act_rows, N = wgt_cols, K = size.
        m = int(self.algo_counts["act_rows"])
        n = int(self.algo_counts["wgt_cols"])
        k = int(self.size)
        return 2 * m * n * k

    def algo_bytes(self) -> int:
        # Count algorithmic bytes for A, B, and C based on streamed rows/cols.
        elem_bytes = self._dtype_bytes()
        m = int(self.algo_counts["act_rows"])
        n = int(self.algo_counts["wgt_cols"])
        return (m * self.size + n * self.size + m * self.size) * elem_bytes

    def algo_arithmetic_intensity(self) -> float:
        bytes_total = self.algo_bytes()
        return (self.algo_flops() / bytes_total) if bytes_total else 0.0

    def internal_bytes_total(self) -> int:
        return (
            int(self.internal_bytes["weight_shift"])
            + int(self.internal_bytes["act_shift"])
            + int(self.internal_bytes["psum_shift"])
            + int(self.internal_bytes["output"])
        )

    def internal_bytes_valid_total(self) -> int:
        return (
            int(self.internal_bytes_valid["weight_shift"])
            + int(self.internal_bytes_valid["act_shift"])
            + int(self.internal_bytes_valid["psum_shift"])
            + int(self.internal_bytes_valid["output"])
        )

    def enqueue(self, activations: List[float], dtype: Optional[object] = None, *, count_algo: bool = True) -> bool:
        """Enqueue one activation vector into input FIFO (left boundary), one value per row."""
        if len(activations) != self.size:
            raise ValueError("activations must match systolic array size")

        if any(q.is_full() for q in self._input_fifo_left):
            return False

        dtype = self._resolve_dtype(dtype)
        for i in range(self.size):
            self._input_fifo_left[i].enqueue(cast_scalar(activations[i], dtype))
        if count_algo:
            self.algo_counts["act_rows"] += 1
            self._algo_out_pending += 1
        return True

    def enqueue_weights(
        self, weights: List[float], dtype: Optional[object] = None, *, count_algo: bool = True
    ) -> bool:
        """Enqueue one weight vector to preload row-stationary weights through shared pass buses."""
        if len(weights) != self.size:
            raise ValueError("weights must match systolic array size")
        if any(q.is_full() for q in self._weight_boundary):
            return False
        dtype_norm = self._resolve_dtype(dtype)
        for i in range(self.size):
            self._weight_boundary[i].enqueue(cast_scalar(weights[i], dtype_norm))
        if count_algo:
            self.algo_counts["wgt_cols"] += 1
        return True

    def enqueue_psums(self, psums: List[float], dtype: Optional[object] = None) -> bool:
        """Optional psum input FIFO injection (top boundary), one value per column."""
        if len(psums) != self.size:
            raise ValueError("psums must match systolic array size")
        if any(q.is_full() for q in self._psum_input_fifo_top):
            return False
        dtype_norm = self._resolve_dtype(dtype)
        for j in range(self.size):
            self._psum_input_fifo_top[j].enqueue(cast_scalar(psums[j], dtype_norm))
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
        dtype_norm = self._resolve_dtype(None)
        for i in range(self.size):
            for j in range(self.size):
                self.array[i][j].weight = cast_scalar(weights[i][j], dtype_norm)

    def tick(self, time: Optional[float] = None) -> None:
        # Stall freezes forward progress so values/ready can be preserved for lossless backpressure.
        if self.stall:
            return

        old_act = [[self.array[i][j].activation_latch for j in range(self.size)] for i in range(self.size)]
        old_w = [[self.array[i][j].weight for j in range(self.size)] for i in range(self.size)]
        old_acc = [[self.array[i][j].accumulation for j in range(self.size)] for i in range(self.size)]
        old_mul = [[self.array[i][j].mul_reg for j in range(self.size)] for i in range(self.size)]
        if self.start:
            any_active = False
            for i in range(self.size):
                for j in range(self.size):
                    if old_act[i][j] != 0.0 and old_w[i][j] != 0.0:
                        any_active = True
                        break
                if any_active:
                    break
            if any_active:
                self.valid_mac_cycles += 1

        # Shared rightward pass bus is muxed between weight preload and activation shift.
        if self.weight_en:
            elem_bytes = self._dtype_bytes()
            if self.start:
                self.internal_bytes["weight_shift"] += self.size * self.size * elem_bytes
            for i in range(self.size):
                in_val = self._weight_boundary[i].dequeue()
                pass_bus = float(in_val) if in_val is not None else 0.0
                for j in range(self.size):
                    prev_weight = old_w[i][j]
                    self.array[i][j].weight = pass_bus
                    if self.start and pass_bus != 0.0:
                        self.internal_bytes_valid["weight_shift"] += elem_bytes
                    pass_bus = prev_weight
        elif self.mac_shift:
            elem_bytes = self._dtype_bytes()
            if self.start:
                self.internal_bytes["act_shift"] += self.size * self.size * elem_bytes
            for i in range(self.size):
                in_val = self._input_fifo_left[i].dequeue()
                pass_bus = float(in_val) if in_val is not None else 0.0
                for j in range(self.size):
                    prev_act = old_act[i][j]
                    self.array[i][j].activation_latch = pass_bus
                    if self.start and pass_bus != 0.0:
                        self.internal_bytes_valid["act_shift"] += elem_bytes
                    pass_bus = prev_act

        # Stage 2: add registered product with vertical psum input.
        elem_bytes = self._dtype_bytes()
        if self.start:
            self.internal_bytes["psum_shift"] += self.size * self.size * elem_bytes
        for i in range(self.size):
            for j in range(self.size):
                top_boundary = self._psum_input_fifo_top[j].dequeue() if i == 0 else None
                psum_in = float(top_boundary) if top_boundary is not None else (0.0 if i == 0 else old_acc[i - 1][j])
                if self.start and psum_in != 0.0:
                    self.internal_bytes_valid["psum_shift"] += elem_bytes
                acc = old_mul[i][j] + psum_in
                self.array[i][j].count_add(is_psum=(i == 0))
                self.metrics["add_ops"] += 1
                if i == 0:
                    self.metrics["psum_adds"] += 1
                if self._current_dtype is not None:
                    cast_acc = cast_scalar(acc, self._current_dtype)
                    self._note_cast(acc, cast_acc)
                    acc = cast_acc
                self.array[i][j].accumulation = float(acc)

        # Stage 1: combinational multiply, product captured into register for next cycle.
        for i in range(self.size):
            for j in range(self.size):
                if self.start:
                    prod = self.array[i][j].activation_latch * self.array[i][j].weight
                    self.array[i][j].count_mul()
                    self.metrics["mul_ops"] += 1
                    self.metrics["mac_ops"] += 1
                    if self._current_dtype is not None:
                        cast_prod = cast_scalar(prod, self._current_dtype)
                        self._note_cast(prod, cast_prod)
                        prod = cast_prod
                    self.array[i][j].mul_reg = float(prod)

        # value_ready is asserted two cycles after start.
        self.value_ready = self._start_pipe_1
        self._start_pipe_1 = self._start_pipe_0
        self._start_pipe_0 = self.start

        if self.value_ready:
            bottom_row = [self.array[self.size - 1][j].accumulation for j in range(self.size)]
            self.psum_output_fifo_bottom.append(bottom_row)
            if self._algo_out_pending > 0:
                self.algo_counts["out_rows"] += 1
                self._algo_out_pending -= 1
            self.internal_bytes["output"] += self.size * self._dtype_bytes()
            if self.start:
                elem_bytes = self._dtype_bytes()
                for val in bottom_row:
                    if val != 0.0:
                        self.internal_bytes_valid["output"] += elem_bytes

    def get_buffer(self) -> List[List[float]]:
        return self.psum_output_fifo_bottom

    def boundary_levels(self) -> List[int]:
        """Current fill level of each input FIFO (left boundary)."""
        return [len(q) for q in self._input_fifo_left]
