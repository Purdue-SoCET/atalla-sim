from typing import List, Optional
import math

from base.dtype import DType, cast_scalar, normalize_dtype
from base.clocked_object import Clocked
from base.queue import SimQueue


class TPUCell4Input:
    """Grouped TPU-style MAC cell.

    Each cell stores a vector of GROUP_SIZE weights, accepts a vector of
    GROUP_SIZE activations, computes a grouped dot product, then adds a psum.
    This mirrors the TPU RTL structure more closely than the old scalar PE mesh.
    """

    def __init__(self, group_size: int = 4):
        self.group_size = int(group_size)
        self.activation_latch: List[float] = [0.0] * self.group_size
        self.weight: List[float] = [0.0] * self.group_size
        self.accumulation: float = 0.0
        self.mul_reg: float = 0.0
        self.mul_ops: int = 0
        self.add_ops: int = 0
        self.psum_adds: int = 0
        self.mac_ops: int = 0

    def _input(self, activation: List[float]) -> None:
        vals = list(activation)[: self.group_size]
        if len(vals) < self.group_size:
            vals += [0.0] * (self.group_size - len(vals))
        self.activation_latch = [float(v) for v in vals]

    def _weight(self, weight: List[float]) -> None:
        vals = list(weight)[: self.group_size]
        if len(vals) < self.group_size:
            vals += [0.0] * (self.group_size - len(vals))
        self.weight = [float(v) for v in vals]

    def _accumulation(self, accumulation: float) -> None:
        self.accumulation = float(accumulation)

    def count_mul(self, count: int) -> None:
        self.mul_ops += int(count)
        self.mac_ops += int(count)

    def count_add(self, count: int = 1, is_psum: bool = False) -> None:
        self.add_ops += int(count)
        if is_psum:
            self.psum_adds += int(count)


class SystolicArrayTPU(Clocked):
    def __init__(
        self,
        size: int,
        boundary_buffer_depth: int = 32,
        dtype: Optional[object] = None,
        *,
        group_size: int = 4,
        mul_latency: int = 2,
        add4_latency: int = 3,
        add2_latency: int = 1,
    ):
        super().__init__()
        self.size = int(size)
        self.group_size = int(group_size)
        self.num_groups = max(1, math.ceil(self.size / self.group_size))
        self.mul_latency = max(1, int(mul_latency))
        self.add4_latency = max(1, int(add4_latency))
        self.add2_latency = max(1, int(add2_latency))
        self.pipeline_latency = self.mul_latency + self.add4_latency + self.add2_latency
        self.output_latency = max(
            1,
            (self.size - 1) + self.mul_latency + self.add4_latency + (self.num_groups * self.add2_latency),
        )
        self.dtype = normalize_dtype(dtype, default=None)
        self._current_dtype: Optional[DType] = None
        self.algo_counts = {"act_rows": 0, "wgt_cols": 0, "out_rows": 0}
        self.metrics = {"mul_ops": 0, "add_ops": 0, "psum_adds": 0, "mac_ops": 0}
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
        self.active_pe_sum: int = 0
        self.compute_window_cycles: int = 0
        self.compute_window_active_pe_sum: int = 0
        self.max_active_pes_in_cycle: int = 0
        self.saturation_count: int = 0
        self.overflow_count: int = 0
        self.psum_output_fifo_bottom: List[List[float]] = []
        self.array: List[List[TPUCell4Input]] = [
            [TPUCell4Input(self.group_size) for _ in range(self.size)] for _ in range(self.num_groups)
        ]
        self._input_fifo_left: List[SimQueue[List[float]]] = [
            SimQueue(boundary_buffer_depth) for _ in range(self.num_groups)
        ]
        self._input_algo_flags: SimQueue[bool] = SimQueue(boundary_buffer_depth)
        self._weight_boundary: List[SimQueue[List[float]]] = [
            SimQueue(boundary_buffer_depth) for _ in range(self.num_groups)
        ]
        self._psum_input_fifo_top: List[SimQueue[float]] = [SimQueue(boundary_buffer_depth) for _ in range(self.size)]
        self._algo_out_pending: int = 0

        self.weight_en: bool = False
        self.mac_shift: bool = True
        self.start: bool = False
        self.stall: bool = False
        self.value_ready: bool = False
        self._output_pipe: List[Optional[List[float]]] = [None] * self.output_latency
        self._output_tag_pipe: List[bool] = [False] * self.output_latency

    def _resolve_dtype(self, dtype: Optional[object]) -> DType:
        dtype_norm = normalize_dtype(dtype, default=self.dtype)
        if dtype_norm is None:
            raise ValueError("dtype must be specified")
        if self._current_dtype is None:
            self._current_dtype = dtype_norm
        if dtype_norm != self._current_dtype:
            raise ValueError("tpu dtype mismatch: job=%s req=%s" % (self._current_dtype, dtype_norm))
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

    def _split_groups(self, values: List[float]) -> List[List[float]]:
        out = []
        vals = list(values)
        for g in range(self.num_groups):
            start = g * self.group_size
            chunk = vals[start : start + self.group_size]
            if len(chunk) < self.group_size:
                chunk += [0.0] * (self.group_size - len(chunk))
            out.append(chunk)
        return out

    def algo_flops(self) -> int:
        m = int(self.algo_counts["act_rows"])
        n = int(self.algo_counts["wgt_cols"])
        k = int(self.size)
        return 2 * m * n * k

    def algo_bytes(self) -> int:
        elem_bytes = self._dtype_bytes()
        m = int(self.algo_counts["act_rows"])
        n = int(self.algo_counts["wgt_cols"])
        return (m * self.size + n * self.size + m * self.size) * elem_bytes

    def algo_arithmetic_intensity(self) -> float:
        bytes_total = self.algo_bytes()
        return (self.algo_flops() / bytes_total) if bytes_total else 0.0

    def internal_bytes_total(self) -> int:
        return sum(int(v) for v in self.internal_bytes.values())

    def internal_bytes_valid_total(self) -> int:
        return sum(int(v) for v in self.internal_bytes_valid.values())

    def warmup_cycles(self) -> int:
        return 0

    def flush_cycles(self) -> int:
        return self.output_latency

    def active_lane_count(self) -> int:
        active_lanes = 0
        for g in range(self.num_groups):
            for j in range(self.size):
                for lane in range(self.group_size):
                    if self.array[g][j].activation_latch[lane] != 0.0 and self.array[g][j].weight[lane] != 0.0:
                        active_lanes += 1
        return active_lanes

    def active_cell_count(self) -> int:
        active_cells = 0
        for g in range(self.num_groups):
            for j in range(self.size):
                if any(
                    self.array[g][j].activation_latch[lane] != 0.0 and self.array[g][j].weight[lane] != 0.0
                    for lane in range(self.group_size)
                ):
                    active_cells += 1
        return active_cells

    def enqueue(self, activations: List[float], dtype: Optional[object] = None, *, count_algo: bool = True) -> bool:
        if len(activations) != self.size:
            raise ValueError("activations must match systolic array size")
        if any(q.is_full() for q in self._input_fifo_left) or self._input_algo_flags.is_full():
            return False
        dtype_norm = self._resolve_dtype(dtype)
        groups = self._split_groups(activations)
        for idx, group in enumerate(groups):
            cast_group = [cast_scalar(v, dtype_norm) for v in group]
            self._input_fifo_left[idx].enqueue(cast_group)
        self._input_algo_flags.enqueue(bool(count_algo))
        if count_algo:
            self.algo_counts["act_rows"] += 1
            self._algo_out_pending += 1
        return True

    def enqueue_weights(self, weights: List[float], dtype: Optional[object] = None, *, count_algo: bool = True) -> bool:
        if len(weights) != self.size:
            raise ValueError("weights must match systolic array size")
        if any(q.is_full() for q in self._weight_boundary):
            return False
        dtype_norm = self._resolve_dtype(dtype)
        groups = self._split_groups(weights)
        for idx, group in enumerate(groups):
            cast_group = [cast_scalar(v, dtype_norm) for v in group]
            self._weight_boundary[idx].enqueue(cast_group)
        if count_algo:
            self.algo_counts["wgt_cols"] += 1
        return True

    def enqueue_psums(self, psums: List[float], dtype: Optional[object] = None) -> bool:
        if len(psums) != self.size:
            raise ValueError("psums must match systolic array size")
        if any(q.is_full() for q in self._psum_input_fifo_top):
            return False
        dtype_norm = self._resolve_dtype(dtype)
        for j in range(self.size):
            self._psum_input_fifo_top[j].enqueue(cast_scalar(psums[j], dtype_norm))
        return True

    def set_control(self, *, weight_en: Optional[bool] = None, mac_shift: Optional[bool] = None, start: Optional[bool] = None, stall: Optional[bool] = None) -> None:
        if weight_en is not None:
            self.weight_en = bool(weight_en)
        if mac_shift is not None:
            self.mac_shift = bool(mac_shift)
        if start is not None:
            self.start = bool(start)
        if stall is not None:
            self.stall = bool(stall)

    def load_weights(self, weights: List[List[float]]) -> None:
        if len(weights) != self.size or any(len(row) != self.size for row in weights):
            raise ValueError("weights must be size x size")
        dtype_norm = self._resolve_dtype(None)
        for col in range(self.size):
            col_vec = [weights[row][col] for row in range(self.size)]
            groups = self._split_groups(col_vec)
            for g in range(self.num_groups):
                self.array[g][col].weight = [cast_scalar(v, dtype_norm) for v in groups[g]]

    def tick(self, time: Optional[float] = None) -> None:
        if self._consume_tick(time) is None:
            return

        if self.stall:
            return

        old_act = [[list(self.array[g][j].activation_latch) for j in range(self.size)] for g in range(self.num_groups)]
        old_w = [[list(self.array[g][j].weight) for j in range(self.size)] for g in range(self.num_groups)]
        old_acc = [[self.array[g][j].accumulation for j in range(self.size)] for g in range(self.num_groups)]
        old_mul = [[self.array[g][j].mul_reg for j in range(self.size)] for g in range(self.num_groups)]

        issued_groups = [[0.0] * self.group_size for _ in range(self.num_groups)]
        issued_is_algo = False

        if self.start:
            active_pes = 0
            for g in range(self.num_groups):
                for j in range(self.size):
                    for lane in range(self.group_size):
                        if old_act[g][j][lane] != 0.0 and old_w[g][j][lane] != 0.0:
                            active_pes += 1
            self.compute_window_cycles += 1
            self.compute_window_active_pe_sum += active_pes
            if active_pes > self.max_active_pes_in_cycle:
                self.max_active_pes_in_cycle = active_pes
            if active_pes > 0:
                self.valid_mac_cycles += 1
                self.active_pe_sum += active_pes

        if self.weight_en:
            elem_bytes = self._dtype_bytes()
            self.internal_bytes["weight_shift"] += self.size * self.size * elem_bytes
            for g in range(self.num_groups):
                in_val = self._weight_boundary[g].dequeue()
                pass_bus = list(in_val) if in_val is not None else [0.0] * self.group_size
                for j in range(self.size):
                    prev_weight = old_w[g][j]
                    self.array[g][j].weight = [float(v) for v in pass_bus]
                    if any(v != 0.0 for v in pass_bus):
                        self.internal_bytes_valid["weight_shift"] += sum(1 for v in pass_bus if v != 0.0) * elem_bytes
                    pass_bus = prev_weight
        elif self.mac_shift:
            elem_bytes = self._dtype_bytes()
            self.internal_bytes["act_shift"] += self.size * self.size * elem_bytes
            flag = self._input_algo_flags.dequeue()
            issued_is_algo = bool(flag)
            for g in range(self.num_groups):
                in_val = self._input_fifo_left[g].dequeue()
                pass_bus = list(in_val) if in_val is not None else [0.0] * self.group_size
                issued_groups[g] = list(pass_bus)
                for j in range(self.size):
                    prev_act = old_act[g][j]
                    self.array[g][j].activation_latch = [float(v) for v in pass_bus]
                    if any(v != 0.0 for v in pass_bus):
                        self.internal_bytes_valid["act_shift"] += sum(1 for v in pass_bus if v != 0.0) * elem_bytes
                    pass_bus = prev_act

        elem_bytes = self._dtype_bytes()
        self.internal_bytes["psum_shift"] += self.num_groups * self.size * elem_bytes
        for g in range(self.num_groups):
            for j in range(self.size):
                top_boundary = self._psum_input_fifo_top[j].dequeue() if g == 0 else None
                psum_in = float(top_boundary) if top_boundary is not None else (0.0 if g == 0 else old_acc[g - 1][j])
                if psum_in != 0.0:
                    self.internal_bytes_valid["psum_shift"] += elem_bytes
                acc = old_mul[g][j] + psum_in
                self.array[g][j].count_add(1, is_psum=True)
                self.metrics["add_ops"] += 1
                self.metrics["psum_adds"] += 1
                if self._current_dtype is not None:
                    cast_acc = cast_scalar(acc, self._current_dtype)
                    self._note_cast(acc, cast_acc)
                    acc = cast_acc
                self.array[g][j].accumulation = float(acc)

        for g in range(self.num_groups):
            for j in range(self.size):
                if self.start:
                    products = [old_act[g][j][lane] * old_w[g][j][lane] for lane in range(self.group_size)]
                    dot = sum(products)
                    mul_count = self.group_size
                    add_count = max(0, self.group_size - 1)
                    self.array[g][j].count_mul(mul_count)
                    self.array[g][j].count_add(add_count, is_psum=False)
                    self.metrics["mul_ops"] += mul_count
                    self.metrics["mac_ops"] += mul_count
                    self.metrics["add_ops"] += add_count
                    if self._current_dtype is not None:
                        cast_dot = cast_scalar(dot, self._current_dtype)
                        self._note_cast(dot, cast_dot)
                        dot = cast_dot
                    self.array[g][j].mul_reg = float(dot)

        issued_output = None
        if self.start:
            full_input = [float(v) for group in issued_groups for v in group][: self.size]
            issued_output = []
            for j in range(self.size):
                col_sum = 0.0
                for g in range(self.num_groups):
                    group_dot = 0.0
                    for lane in range(self.group_size):
                        group_dot += float(issued_groups[g][lane]) * float(old_w[g][j][lane])
                    col_sum += group_dot
                if self._current_dtype is not None:
                    cast_sum = cast_scalar(col_sum, self._current_dtype)
                    self._note_cast(col_sum, cast_sum)
                    col_sum = cast_sum
                issued_output.append(float(col_sum))

        ready_row = self._output_pipe[-1]
        ready_tag = self._output_tag_pipe[-1]
        self._output_pipe = [issued_output] + self._output_pipe[:-1]
        self._output_tag_pipe = [issued_is_algo] + self._output_tag_pipe[:-1]
        self.value_ready = bool(ready_tag and ready_row is not None)

        if self.value_ready and ready_row is not None:
            self.psum_output_fifo_bottom.append(list(ready_row))
            if self._algo_out_pending > 0:
                self.algo_counts["out_rows"] += 1
                self._algo_out_pending -= 1
            self.internal_bytes["output"] += self.size * self._dtype_bytes()
            for val in ready_row:
                if val != 0.0:
                    self.internal_bytes_valid["output"] += elem_bytes

    def get_buffer(self) -> List[List[float]]:
        return self.psum_output_fifo_bottom

    def boundary_levels(self) -> List[int]:
        return [len(q) for q in self._input_fifo_left]
