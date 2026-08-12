from typing import List, Optional
import math

import numpy as np

from base.dtype import DType, cast_scalar, cast_vector, normalize_dtype
from base.clocked_object import Clocked
from base.queue import SimQueue

from native import kernels as _native


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


class _ArrayCell:
    """A cell of a live SystolicArrayTPU, viewing the array's flat buffers.

    The datapath state lives in contiguous float64 arrays so the native kernel
    can work on it without marshalling. This class keeps the old per-cell
    attribute interface working -- `array[g][j].weight`, `.mul_reg`, and so on
    -- by reading and writing straight through to those buffers.

    Op counters are not stored per cell. Every cell is charged the identical
    amount on every tick (one psum add always, plus group_size multiplies and
    group_size-1 adds while `start` is asserted), so the totals are held once
    on the owning array and shared by all cells. A cell only gets its own entry
    if something calls `count_mul`/`count_add` on it directly.
    """

    __slots__ = ("_o", "_g", "_j")

    def __init__(self, owner: "SystolicArrayTPU", g: int, j: int):
        self._o = owner
        self._g = g
        self._j = j

    @property
    def group_size(self) -> int:
        return self._o.group_size

    @property
    def activation_latch(self) -> List[float]:
        return self._o._arr.act[self._g, :, self._j].tolist()

    @activation_latch.setter
    def activation_latch(self, values) -> None:
        self._o._arr.act[self._g, :, self._j] = self._o._fit(values)

    @property
    def weight(self) -> List[float]:
        return self._o._arr.wgt[self._g, :, self._j].tolist()

    @weight.setter
    def weight(self, values) -> None:
        self._o._arr.wgt[self._g, :, self._j] = self._o._fit(values)

    @property
    def accumulation(self) -> float:
        return float(self._o._arr.acc[self._g, self._j])

    @accumulation.setter
    def accumulation(self, value: float) -> None:
        self._o._arr.acc[self._g, self._j] = float(value)

    @property
    def mul_reg(self) -> float:
        return float(self._o._arr.mul[self._g, self._j])

    @mul_reg.setter
    def mul_reg(self, value: float) -> None:
        self._o._arr.mul[self._g, self._j] = float(value)

    def _extra(self, key: str) -> int:
        return self._o._cell_extra.get((self._g, self._j, key), 0)

    def _bump(self, key: str, amount: int) -> None:
        k = (self._g, self._j, key)
        self._o._cell_extra[k] = self._o._cell_extra.get(k, 0) + int(amount)

    @property
    def mul_ops(self) -> int:
        return self._o._common_mul_ops + self._extra("mul")

    @property
    def add_ops(self) -> int:
        return self._o._common_add_ops + self._extra("add")

    @property
    def psum_adds(self) -> int:
        return self._o._common_psum_adds + self._extra("psum")

    @property
    def mac_ops(self) -> int:
        return self._o._common_mac_ops + self._extra("mac")

    def _input(self, activation: List[float]) -> None:
        self.activation_latch = activation

    def _weight(self, weight: List[float]) -> None:
        self.weight = weight

    def _accumulation(self, accumulation: float) -> None:
        self.accumulation = accumulation

    def count_mul(self, count: int) -> None:
        self._bump("mul", count)
        self._bump("mac", count)

    def count_add(self, count: int = 1, is_psum: bool = False) -> None:
        self._bump("add", count)
        if is_psum:
            self._bump("psum", count)


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

        # Flat datapath state, shared by the native kernel and the cell views.
        self._arr = _native.SaArrays(self.num_groups, self.size, self.group_size)
        self._zero_group = [0.0] * self.group_size
        self._issued_nonzero = False
        self._common_mul_ops = 0
        self._common_add_ops = 0
        self._common_psum_adds = 0
        self._common_mac_ops = 0
        self._cell_extra = {}
        self.array: List[List[_ArrayCell]] = [
            [_ArrayCell(self, g, j) for j in range(self.size)] for g in range(self.num_groups)
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

    def _fit(self, values) -> List[float]:
        """Trim or zero-pad a lane vector to exactly group_size entries."""
        vals = [float(v) for v in values][: self.group_size]
        if len(vals) < self.group_size:
            vals += [0.0] * (self.group_size - len(vals))
        return vals

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

    def _cast_group(self, values: List[float], dtype: DType) -> List[float]:
        """Cast a whole boundary vector at once.

        cast_vector batches through numpy, which is far cheaper per element
        than a cast_scalar call. INT8 keeps the scalar path because numpy's
        array cast wraps out-of-range values while the scalar cast raises.
        """
        if dtype == DType.INT8:
            return [cast_scalar(v, dtype) for v in values]
        return cast_vector(values, dtype)

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
        return int(np.count_nonzero((self._arr.act != 0.0) & (self._arr.wgt != 0.0)))

    def active_cell_count(self) -> int:
        both = (self._arr.act != 0.0) & (self._arr.wgt != 0.0)
        return int(np.count_nonzero(both.any(axis=1)))

    def enqueue(self, activations: List[float], dtype: Optional[object] = None, *, count_algo: bool = True) -> bool:
        if len(activations) != self.size:
            raise ValueError("activations must match systolic array size")
        if any(q.is_full() for q in self._input_fifo_left) or self._input_algo_flags.is_full():
            return False
        dtype_norm = self._resolve_dtype(dtype)
        cast_all = self._cast_group(list(activations), dtype_norm)
        for idx, group in enumerate(self._split_groups(cast_all)):
            self._input_fifo_left[idx].enqueue(group)
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
        cast_all = self._cast_group(list(weights), dtype_norm)
        for idx, group in enumerate(self._split_groups(cast_all)):
            self._weight_boundary[idx].enqueue(group)
        if count_algo:
            self.algo_counts["wgt_cols"] += 1
        return True

    def enqueue_psums(self, psums: List[float], dtype: Optional[object] = None) -> bool:
        if len(psums) != self.size:
            raise ValueError("psums must match systolic array size")
        if any(q.is_full() for q in self._psum_input_fifo_top):
            return False
        dtype_norm = self._resolve_dtype(dtype)
        cast_all = self._cast_group(list(psums), dtype_norm)
        for j in range(self.size):
            self._psum_input_fifo_top[j].enqueue(cast_all[j])
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
            cast_col = self._cast_group(col_vec, dtype_norm)
            for g, group in enumerate(self._split_groups(cast_col)):
                self._arr.wgt[g, :, col] = group

    # -- cast mode plumbing --------------------------------------------------

    def _cast_mode(self) -> int:
        return _native.CAST_INT8 if self._current_dtype == DType.INT8 else _native.CAST_HALF

    def _fallback_cast(self, arr: np.ndarray):
        """numpy equivalent of the kernel's cast; returns (out, sat, ovf)."""
        dtype = self._current_dtype
        if dtype == DType.INT8:
            out = np.trunc(arr)
            if np.any(~np.isfinite(out)) or np.any((out < -128.0) | (out > 127.0)):
                raise OverflowError("int8 cast out of bounds")
            return out, 0, 0
        with np.errstate(over="ignore", invalid="ignore"):
            out = arr.astype(np.float16).astype(np.float64)
        if dtype == DType.FP16:
            sat = int(np.count_nonzero(np.abs(arr) > 65504.0))
            ovf = int(np.count_nonzero(~np.isfinite(out)))
            return out, sat, ovf
        return out, 0, 0

    def _tick_fallback(self, start: bool, weight_en: bool, mac_shift: bool):
        """Pure-numpy mirror of atalla_sa_tick, used when the .so is absent.

        Phase order, reduction order and the g-descending psum walk match the
        native kernel exactly, so both paths produce identical bits.
        """
        a = self._arr
        G, S, GS = a.G, a.S, a.GS
        has_dtype = self._current_dtype is not None
        active_pes = 0
        psum_nnz = 0
        sat_total = 0
        ovf_total = 0
        shift_nnz = 0

        # Overflow to inf, and the inf-inf NaNs that follow, are ordinary
        # outcomes for this model; the native path reports them without
        # complaint and numpy should not warn about them either.
        with np.errstate(over="ignore", invalid="ignore"):
            # Phase A
            if start:
                active_pes = int(np.count_nonzero((a.act != 0.0) & (a.wgt != 0.0)))

            # Phase B: psum accumulate, g descending so acc[g-1] is still old.
            for g in range(G - 1, -1, -1):
                if g == 0:
                    psum_in = np.where(a.psum_valid.astype(bool), a.psum_top, 0.0)
                else:
                    psum_in = a.acc[g - 1].copy()
                psum_nnz += int(np.count_nonzero(psum_in))
                acc = a.mul[g] + psum_in
                if has_dtype:
                    acc, s, o = self._fallback_cast(acc)
                    sat_total += s
                    ovf_total += o
                a.acc[g] = acc

            # Phase C: grouped MAC, lane by lane in Python's accumulation order.
            if start:
                for g in range(G):
                    dot = np.zeros(S, dtype=np.float64)
                    for lane in range(GS):
                        dot += a.act[g, lane] * a.wgt[g, lane]
                    if has_dtype:
                        dot, s, o = self._fallback_cast(dot)
                        sat_total += s
                        ovf_total += o
                    a.mul[g] = dot

            # Phase D: output row.
            if start:
                total = np.zeros(S, dtype=np.float64)
                issued = a.issued.reshape(G, GS)
                for g in range(G):
                    grp = np.zeros(S, dtype=np.float64)
                    for lane in range(GS):
                        grp += issued[g, lane] * a.wgt[g, lane]
                    total += grp
                if has_dtype:
                    total, s, o = self._fallback_cast(total)
                    sat_total += s
                    ovf_total += o
                a.issued_out[:] = total

            # Phase E: systolic shift by one column.
            if weight_en or mac_shift:
                base = a.wgt if weight_en else a.act
                base[:, :, 1:] = base[:, :, :-1]
                base[:, :, 0] = a.shift_in.reshape(G, GS)
                shift_nnz = int(np.count_nonzero(base))

        a.metrics[_native.SaArrays.M_ACTIVE_PES] = active_pes
        a.metrics[_native.SaArrays.M_PSUM_NNZ] = psum_nnz
        a.metrics[_native.SaArrays.M_SAT] = sat_total
        a.metrics[_native.SaArrays.M_OVF] = ovf_total
        a.metrics[_native.SaArrays.M_SHIFT_NNZ] = shift_nnz
        a.metrics[_native.SaArrays.M_INT8_RANGE] = 0
        return a.metrics

    def tick(self, time: Optional[float] = None) -> None:
        if self._consume_tick(time) is None:
            return

        if self.stall:
            return

        a = self._arr
        G, S, GS = a.G, a.S, a.GS
        weight_en = self.weight_en
        mac_shift = (not weight_en) and self.mac_shift
        start = self.start
        elem_bytes = self._dtype_bytes()
        issued_is_algo = False

        # -- Drain the boundary queues into the kernel's input buffers -------
        # Values are gathered into plain lists and written to each buffer in a
        # single slice assignment; element-at-a-time numpy stores cost more
        # than the kernel call they feed.
        zeros = self._zero_group
        if weight_en:
            self.internal_bytes["weight_shift"] += self.size * self.size * elem_bytes
            vals = []
            for q in self._weight_boundary:
                in_val = q.dequeue()
                vals.extend(zeros if in_val is None else in_val)
            a.shift_in[:] = vals
        elif mac_shift:
            self.internal_bytes["act_shift"] += self.size * self.size * elem_bytes
            issued_is_algo = bool(self._input_algo_flags.dequeue())
            vals = []
            for q in self._input_fifo_left:
                in_val = q.dequeue()
                vals.extend(zeros if in_val is None else in_val)
            a.shift_in[:] = vals
            # The activations entering the array are also the row issued this
            # cycle; on a weight shift no activations are issued at all.
            a.issued[:] = vals
            self._issued_nonzero = True
        if not mac_shift and self._issued_nonzero:
            a.issued.fill(0.0)
            self._issued_nonzero = False

        tops = []
        valids = []
        for q in self._psum_input_fifo_top:
            val = q.dequeue()
            if val is None:
                tops.append(0.0)
                valids.append(0)
            else:
                tops.append(val)
                valids.append(1)
        a.psum_top[:] = tops
        a.psum_valid[:] = valids

        self.internal_bytes["psum_shift"] += G * S * elem_bytes

        # -- Datapath ---------------------------------------------------------
        has_dtype = self._current_dtype is not None
        if _native.HAVE_NATIVE:
            metrics = a.tick(
                start, weight_en, mac_shift,
                has_dtype, self._cast_mode(),
                self._current_dtype == DType.FP16,
            )
            if metrics[_native.SaArrays.M_INT8_RANGE]:
                raise OverflowError("int8 cast out of bounds")
        else:
            metrics = self._tick_fallback(start, weight_en, mac_shift)

        active_pes = int(metrics[_native.SaArrays.M_ACTIVE_PES])
        self.internal_bytes_valid["psum_shift"] += int(metrics[_native.SaArrays.M_PSUM_NNZ]) * elem_bytes
        self.saturation_count += int(metrics[_native.SaArrays.M_SAT])
        self.overflow_count += int(metrics[_native.SaArrays.M_OVF])
        if weight_en:
            self.internal_bytes_valid["weight_shift"] += int(metrics[_native.SaArrays.M_SHIFT_NNZ]) * elem_bytes
        elif mac_shift:
            self.internal_bytes_valid["act_shift"] += int(metrics[_native.SaArrays.M_SHIFT_NNZ]) * elem_bytes

        # -- Op counters ------------------------------------------------------
        # Every cell is charged identically, so these are tracked once for the
        # whole array and surfaced per cell by _ArrayCell.
        cells = G * S
        self._common_add_ops += 1
        self._common_psum_adds += 1
        self.metrics["add_ops"] += cells
        self.metrics["psum_adds"] += cells
        if start:
            add_count = max(0, GS - 1)
            self._common_mul_ops += GS
            self._common_mac_ops += GS
            self._common_add_ops += add_count
            self.metrics["mul_ops"] += cells * GS
            self.metrics["mac_ops"] += cells * GS
            self.metrics["add_ops"] += cells * add_count

            self.compute_window_cycles += 1
            self.compute_window_active_pe_sum += active_pes
            if active_pes > self.max_active_pes_in_cycle:
                self.max_active_pes_in_cycle = active_pes
            if active_pes > 0:
                self.valid_mac_cycles += 1
                self.active_pe_sum += active_pes

        issued_output = a.issued_out.tolist() if start else None

        # -- Output pipeline ---------------------------------------------------
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
