"""MEISSA systolic array: a multiplier grid with per-column adder trees.

A standard systolic array interleaves multiply and accumulate in one MAC cell.
MEISSA splits them: a grid of plain multipliers, and one pipelined adder tree
per column that reduces that column's N products to a single value.

    a[0] ->[x]->[x]->[x]->[x]      one column per cycle, left to right
    a[1] ->[x]->[x]->[x]->[x]
    a[2] ->[x]->[x]->[x]->[x]
    a[3] ->[x]->[x]->[x]->[x]
           |    |    |    |
          tree tree tree tree      N products -> 1 sum, pipelined
           |    |    |    |
          red  red  red  red       FP32 -> BF16, before the buffer so it
           |    |    |    |        stores 16-bit values, not 32-bit
          ---- output buffer ----  one bank per column, de-skews
                   |
                  GSAU

Because product generation and reduction are both pipelined, inputs need no
temporal skewing on the way in -- a whole vector is injected at once. The skew
appears on the way *out* instead: an activation reaches column j on cycle j, so
column j finishes j cycles after column 0. The output buffer removes it by
giving each column its own bank with its own write pointer; a read takes one
address across all banks, which is one complete result vector.

Once the pipeline is full the array retires one output vector per cycle.

Precision follows the RTL's BF16 configuration: inputs are BF16, products and
every adder-tree sum are FP32, and a reducer converts to BF16 after the tree.

The adder tree has two shapes, chosen by `use_mixed_adder`:

    pure 2-input    log2(N) levels
    mixed 4/2       log4(N) levels of 4-input adders, plus one 2-input level
                    when N is not a power of four (32 needs one, 16 does not)

Four-input adders are the cheaper way to reduce -- about 2.4x the area
efficiency of three cascaded two-input adders -- but they also round once
instead of three times, so the two shapes do not produce identical values.
Both are modelled.

API matches SystolicArrayTPU, so this drops into GSAUTPUBridge unchanged.

One caller-visible consequence of the hardware: weights arrive through a shift
register, so the FIRST vector pushed ends up in the LAST column. To land
column j of a weight matrix W in array column j, push the columns of W in
reverse. Get it wrong and the array computes A @ W.T with its columns
reversed, silently -- there is nothing in the interface that can catch it.

Transcribed from sysarr_MEISSA_top.sv, mul_grid.sv, pipelined_adder_tree.sv,
mixed_pipelined_adder_tree.sv and output_buffer.sv on the atalla repo's
systolic_array_arch branch.
"""

import ctypes
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from base.clocked_object import Clocked
from base.dtype import DType, bf16_round, cast_vector, normalize_dtype, numpy_dtype
from native import kernels as _native

Time = float


def is_power_of_four(n: int) -> bool:
    """sysarr_MEISSA_top.sv: LOG4_IS_WHOLE -- a power of two whose set bit sits
    at an even position. 16 yes, 32 no."""
    return n > 0 and (n & (n - 1)) == 0 and (n & 0x55555555) != 0


def tree_levels(size: int, use_mixed_adder: bool) -> Dict[str, int]:
    """Levels of each adder kind for one column's reduction."""
    log2n = int(math.ceil(math.log2(max(1, size))))
    if not use_mixed_adder:
        return {"add4": 0, "add2": log2n}
    if is_power_of_four(size):
        return {"add4": (log2n + 1) // 2, "add2": 0}
    # One 2-input level mops up what the 4-input levels cannot.
    return {"add4": (log2n - 1) // 2, "add2": 1}


class SystolicArrayMEISSA(Clocked):
    """Multiplier grid, per-column adder trees, reducers and an output buffer."""

    def __init__(
        self,
        size: int,
        dtype: Optional[object] = None,
        *,
        mul_latency: int = 1,
        add2_latency: int = 1,
        add4_latency: int = 3,
        use_mixed_adder: bool = False,
        output_depth: Optional[int] = None,
        collect_stats: bool = False,
        use_native: bool = True,
    ):
        super().__init__()
        self.size = int(size)
        if self.size <= 0:
            raise ValueError("size must be positive")
        self.mul_latency = max(1, int(mul_latency))
        self.add2_latency = max(1, int(add2_latency))
        self.add4_latency = max(1, int(add4_latency))
        self.use_mixed_adder = bool(use_mixed_adder)
        self.dtype = normalize_dtype(dtype, default=None)

        self.levels = tree_levels(self.size, self.use_mixed_adder)
        #: Cycles from a product being formed to its column's sum leaving the
        #: tree -- the multiplier plus every adder level.
        self.pipeline_depth = (
            self.mul_latency
            + self.levels["add4"] * self.add4_latency
            + self.levels["add2"] * self.add2_latency
        )
        #: output_buffer.sv sizes the banks to cover the whole pipeline.
        self.output_depth = int(output_depth or (self.size + self.pipeline_depth))

        # --- multiplier grid ------------------------------------------------
        # act[i][j] / wgt[i][j]: row i, column j. Activations shift one column
        # per cycle; weights shift only while weight_en is held.
        self._act = np.zeros((self.size, self.size), dtype=np.float32)
        self._wgt = np.zeros((self.size, self.size), dtype=np.float32)
        #: Which injected vector occupies each column; -1 marks a bubble.
        self._col_seq = np.full(self.size, -1, dtype=np.int64)

        # --- adder tree pipelines -------------------------------------------
        # All columns reduce in lockstep, so the delay line is one ring buffer
        # of whole rows rather than N per-column lists: row r of _tree_vals is
        # the sums that entered the trees r cycles ago, -1 in _tree_seq marking
        # a column that was carrying a bubble.
        self._tree_vals = np.zeros((self.pipeline_depth, self.size), dtype=np.float32)
        self._tree_seq = np.full((self.pipeline_depth, self.size), -1, dtype=np.int64)
        self._tree_head = 0

        # --- output buffer ---------------------------------------------------
        # One bank per column, each a ring of (seq, value). A result vector is
        # complete when every bank holds it, which is what de-skews the
        # columns. Rings rather than lists so the kernel can share the storage.
        self._bank_vals = np.zeros((self.size, self.output_depth), dtype=np.float32)
        self._bank_seq = np.full((self.size, self.output_depth), -1, dtype=np.int64)
        self._bank_head = np.zeros(self.size, dtype=np.int32)
        self._bank_count = np.zeros(self.size, dtype=np.int32)
        #: Column indices, kept so the drain does not rebuild an arange every
        #: cycle just to gather one entry per bank.
        self._col_idx = np.arange(self.size)
        self.output_rows: List[List[float]] = []

        # --- control ---------------------------------------------------------
        self.weight_en = False
        self.mac_shift = False
        self.start = False
        self.stall = False
        self._pending_act: Optional[List[float]] = None
        self._pending_wgt: Optional[List[float]] = None
        self._next_seq = 0
        self._retired = 0
        self._tick = -1

        # Credit-based flow control, as in sysarr_MEISSA_top.sv: one credit per
        # vector the pipeline and buffer can still swallow.
        self.max_credits = self.pipeline_depth + self.size - 1
        self.credits = self.max_credits

        self.algo_counts = {"act_rows": 0, "wgt_cols": 0, "out_rows": 0}
        # psum_adds is present and always zero because MEISSA has no partial-sum
        # adder at all -- see INAPPLICABLE_STATS, which names it so a report can
        # say "not applicable" rather than printing a measurement of nothing.
        self.metrics = {"mul_ops": 0, "add_ops": 0, "mac_ops": 0, "psum_adds": 0}
        #: Valid bytes moving inside the array, same keys as SystolicArrayTPU.
        #: psum_shift has no counterpart: nothing accumulates between cells.
        self.internal_bytes = {"weight_shift": 0, "act_shift": 0,
                               "psum_shift": 0, "output": 0}
        self.internal_bytes_valid = {"weight_shift": 0, "act_shift": 0,
                                     "psum_shift": 0, "output": 0}
        self.saturation_count = 0
        self.overflow_count = 0
        self.max_active_pes_in_cycle = 0
        self.compute_window_active_pe_sum = 0
        self.value_ready = False
        self._algo_out_pending = 0
        #: Counting active PEs is two NxN comparisons per cycle for a number
        #: nothing reads unless a sweep asked for it, so it is off by default.
        self.collect_stats = bool(collect_stats)
        self.valid_mac_cycles = 0
        self.compute_window_cycles = 0
        self.active_pe_sum = 0

        self._half = numpy_dtype(self.dtype) if self.dtype is not None else None
        self._state = None
        self.use_native = bool(use_native) and _native.HAVE_NATIVE
        if self.use_native:
            self._state = self._build_native_state()
        # Per-cycle scratch for the batched kernel entry point, grown on demand.
        self._batch = None

    def _build_native_state(self):
        """Point the kernel at the arrays this object already owns.

        Nothing is copied: the struct holds pointers into the same buffers the
        numpy path reads and writes, so the two implementations share one
        state and can be compared cycle for cycle.
        """
        st = _native.MeissaState()
        st.N = self.size
        st.depth = self.pipeline_depth
        st.out_depth = self.output_depth
        st.head = 0
        st.levels4 = self.levels["add4"]
        st.levels2 = self.levels["add2"]
        # 0 = no reducer, 1 = FP16, 2 = BF16 (atalla_meissa_run).
        st.do_reduce = (0 if self.dtype is None
                        else 2 if self.dtype == DType.BF16 else 1)
        st.collect_stats = 1 if self.collect_stats else 0
        st.act = _native.fptr(self._act)
        st.wgt = _native.fptr(self._wgt)
        st.col_seq = _native.i64ptr(self._col_seq)
        st.tree_vals = _native.fptr(self._tree_vals)
        st.tree_seq = _native.i64ptr(self._tree_seq)
        st.bank_vals = _native.fptr(self._bank_vals)
        st.bank_seq = _native.i64ptr(self._bank_seq)
        st.bank_head = _native.i32ptr(self._bank_head)
        st.bank_count = _native.i32ptr(self._bank_count)
        st.next_seq = 0
        st.credits = self.max_credits
        st.max_credits = self.max_credits
        return st

    def _sync_from_native(self) -> None:
        self._tree_head = self._state.head
        self._next_seq = self._state.next_seq
        self.credits = self._state.credits

    def _sync_to_native(self) -> None:
        self._state.head = self._tree_head
        self._state.next_seq = self._next_seq
        self._state.credits = self.credits
        self._state.collect_stats = 1 if self.collect_stats else 0

    # -- geometry ----------------------------------------------------------
    def warmup_cycles(self) -> int:
        """None. The output buffer de-skews, so every row it emits is real."""
        return 0

    def flush_cycles(self) -> int:
        """No flush vectors. A skewed array needs zero rows pushed in behind
        the last real one to shift its results out; MEISSA does not, because
        the grid shifts every cycle whether or not anything is injected and the
        output buffer de-skews. Pushing them would only burn credits.

        For the time the last vector still needs, see drain_cycles().
        """
        return 0

    def drain_cycles(self) -> int:
        """Cycles for the last injected vector to reach the output buffer."""
        return self.size + self.pipeline_depth

    def ready_in(self) -> bool:
        return self.credits > 0

    def can_accept(self) -> bool:
        return self.ready_in() and self._pending_act is None

    # -- ingress -----------------------------------------------------------
    def _cast(self, values: Sequence[float], dtype: Optional[object]) -> List[float]:
        dt = normalize_dtype(dtype, default=self.dtype)
        vals = [float(v) for v in values]
        return cast_vector(vals, dt) if dt is not None else vals

    def enqueue(self, activations: List[float], dtype: Optional[object] = None,
                *, count_algo: bool = True) -> bool:
        """Inject one activation vector into column 0. No input skewing."""
        if len(activations) != self.size:
            raise ValueError("activations must match systolic array size")
        if not self.can_accept():
            return False
        self._pending_act = self._cast(activations, dtype)
        self._pending_count_algo = bool(count_algo)
        if count_algo:
            self.algo_counts["act_rows"] += 1
        return True

    def enqueue_weights(self, weights: List[float], dtype: Optional[object] = None,
                        *, count_algo: bool = True) -> bool:
        """Push one weight vector into column 0, shifting the rest right.

        Weights shift like activations, so the FIRST vector pushed ends up in
        the LAST column: after N pushes, column j holds push number N-1-j.
        """
        if len(weights) != self.size:
            raise ValueError("weights must match systolic array size")
        if self._pending_wgt is not None:
            return False
        self._pending_wgt = self._cast(weights, dtype)
        if count_algo:
            self.algo_counts["wgt_cols"] += 1
        return True

    def enqueue_psums(self, psums: List[float], dtype: Optional[object] = None) -> bool:
        """MEISSA has no partial-sum input: psum injection is disconnected in
        pipelined_adder_tree.sv (sum_out is the tree result alone). Accepted
        only when zero, so a caller that means it gets told."""
        if len(psums) != self.size:
            raise ValueError("psums must match systolic array size")
        if any(float(v) != 0.0 for v in psums):
            raise ValueError(
                "the MEISSA adder tree has no psum input; only zeros accepted")
        return True

    def set_control(self, *, weight_en: Optional[bool] = None,
                    mac_shift: Optional[bool] = None, start: Optional[bool] = None,
                    stall: Optional[bool] = None) -> None:
        if weight_en is not None:
            self.weight_en = bool(weight_en)
        if mac_shift is not None:
            self.mac_shift = bool(mac_shift)
        if start is not None:
            self.start = bool(start)
        if stall is not None:
            self.stall = bool(stall)

    def load_weights(self, weights: List[List[float]]) -> None:
        """Place a whole size x size weight matrix directly, bypassing the
        shift-in. weights[i][j] is row i of column j."""
        if len(weights) != self.size or any(len(r) != self.size for r in weights):
            raise ValueError("weights must be size x size")
        for i, row in enumerate(weights):
            self._wgt[i, :] = np.asarray(self._cast(row, None), dtype=np.float32)

    # -- the adder tree ----------------------------------------------------
    @staticmethod
    def _pad_to(level: np.ndarray, multiple: int) -> np.ndarray:
        """Zero-pad the term axis so a level divides evenly. Adding 0.0 is
        exact, so padding never changes a sum."""
        short = (-level.shape[0]) % multiple
        if not short:
            return level
        return np.concatenate(
            [level, np.zeros((short,) + level.shape[1:], dtype=np.float32)])

    def _reduce_columns(self, products: np.ndarray) -> np.ndarray:
        """Reduce every column at once, in the tree's own shape and order.

        products[i, j] is term i of column j; the result is one sum per column.
        Reducing all columns together is what makes this affordable -- the tree
        is the same for each, so one array operation per level replaces N
        scalar ones, and the pairing and order are untouched.

        Rounding follows the structure, which is the point of the two shapes:
        a 4-input adder rounds ONCE, where three cascaded 2-input adders round
        three times. The 4-input sum is therefore accumulated in double and
        rounded a single time on the way out.
        """
        level = np.asarray(products, dtype=np.float32)

        for _ in range(self.levels["add4"]):
            level = self._pad_to(level, 4)
            wide = level.astype(np.float64)
            level = (wide[0::4] + wide[1::4] + wide[2::4]
                     + wide[3::4]).astype(np.float32)
            self.metrics["add_ops"] += level.size

        for _ in range(self.levels["add2"]):
            level = self._pad_to(level, 2)
            level = level[0::2] + level[1::2]
            self.metrics["add_ops"] += level.size

        while level.shape[0] > 1:          # nothing left for a ragged size
            level = self._pad_to(level, 2)
            level = level[0::2] + level[1::2]
        return level[0]

    def _reduce_to_dtype(self, values: np.ndarray) -> np.ndarray:
        """reducer.sv: FP32 down to the storage type before the output buffer,
        so the banks hold 16-bit values rather than 32-bit ones.

        Done as one numpy round-trip on the whole row. cast_vector would cross
        the ctypes boundary for N values, which costs more than the conversion
        itself; astype rounds to nearest-even identically, which the
        equivalence test pins.
        """
        if self._half is None:
            return np.asarray(values, dtype=np.float32)
        if self.dtype == DType.BF16:
            return bf16_round(values)
        with np.errstate(over="ignore", invalid="ignore"):
            return np.asarray(values, dtype=self._half).astype(np.float32)

    # -- the native path ---------------------------------------------------
    def _ensure_batch(self, cap: int) -> None:
        """Preallocate the planes the kernel reads and writes."""
        if self._batch is not None and self._batch["cap"] >= cap:
            return
        n = self.size
        self._batch = {
            "cap": cap,
            "inject": np.zeros((cap, n), dtype=np.float32),
            "inject_valid": np.zeros(cap, dtype=np.uint8),
            "wgt_push": np.zeros((cap, n), dtype=np.float32),
            "wgt_en": np.zeros(cap, dtype=np.uint8),
            "out_rows": np.zeros((cap, n), dtype=np.float32),
            "out_count": np.zeros(1, dtype=np.int32),
            "metrics": np.zeros(_native.MM_COUNT, dtype=np.int64),
        }

    def _run_native(self, cycles: int) -> int:
        """Hand `cycles` cycles to the kernel in one call.

        This is where the batching pays: the per-cycle cost of the numpy path
        is roughly two dozen small array operations, each carrying interpreter
        and dispatch overhead far larger than the arithmetic. One call covering
        K cycles amortises all of it.
        """
        b = self._batch
        self._sync_to_native()
        _native.lib.atalla_meissa_run(
            ctypes.byref(self._state), ctypes.c_int32(cycles),
            _native.fptr(b["inject"]), _native.u8ptr(b["inject_valid"]),
            _native.fptr(b["wgt_push"]), _native.u8ptr(b["wgt_en"]),
            _native.fptr(b["out_rows"]), _native.i32ptr(b["out_count"]),
            _native.i64ptr(b["metrics"]))
        self._sync_from_native()

        produced = int(b["out_count"][0])
        for r in range(produced):
            self._emit([float(v) for v in b["out_rows"][r]])
        self._absorb_metrics(b["metrics"])
        self.compute_window_cycles += cycles
        self.value_ready = produced > 0
        return produced

    def run_cycles(self, cycles: int, activations: Optional[Sequence] = None,
                   weights: Optional[Sequence] = None) -> int:
        """Advance `cycles` cycles in one go, returning the rows produced.

        `activations[c]` / `weights[c]` are what to inject on cycle c, or None
        for a bubble. Use this wherever the array can run unattended -- a tile
        streamed in one burst, or the drain after the last vector. The
        per-cycle tick() contract still works and is bit-identical; this just
        stops paying Python for every cycle.
        """
        cycles = int(cycles)
        if cycles <= 0:
            return 0
        self._ensure_batch(cycles)
        b = self._batch
        b["inject_valid"][:cycles] = 0
        b["wgt_en"][:cycles] = 0
        if activations:
            for c, row in enumerate(activations[:cycles]):
                if row is not None:
                    b["inject"][c] = np.asarray(self._cast(row, None), dtype=np.float32)
                    b["inject_valid"][c] = 1
                    self.algo_counts["act_rows"] += 1
        if weights:
            for c, row in enumerate(weights[:cycles]):
                if row is not None:
                    b["wgt_push"][c] = np.asarray(self._cast(row, None), dtype=np.float32)
                    b["wgt_en"][c] = 1
                    self.algo_counts["wgt_cols"] += 1
        if not self.use_native:
            return self._run_reference(cycles, b)
        self._tick = (self._tick if self._tick >= 0 else 0) + cycles
        return self._run_native(cycles)

    def _run_reference(self, cycles: int, b) -> int:
        """The numpy path driving the same batch, for equivalence testing."""
        before = len(self.output_rows)
        for c in range(cycles):
            if b["wgt_en"][c]:
                self._pending_wgt = list(b["wgt_push"][c])
                self.weight_en = True
            else:
                self.weight_en = False
            if b["inject_valid"][c]:
                self._pending_act = list(b["inject"][c])
            self._cycle_reference()
        return len(self.output_rows) - before

    def _cycle_reference(self) -> None:
        self._shift_weights()
        self._shift_activations()
        self._advance_trees()
        self._drain_output_buffer()
        self._count_activity()

    # -- one cycle ---------------------------------------------------------
    def tick(self, time: Optional[Time] = None) -> None:
        cycle = self._consume_tick(time, attr_name="_tick")
        if cycle is None or self.stall:
            return

        if self.use_native:
            self._ensure_batch(1)
            b = self._batch
            b["inject_valid"][0] = 0
            b["wgt_en"][0] = 0
            if self._pending_act is not None:
                b["inject"][0] = np.asarray(self._pending_act, dtype=np.float32)
                b["inject_valid"][0] = 1
                self._pending_act = None
            if self.weight_en and self._pending_wgt is not None:
                b["wgt_push"][0] = np.asarray(self._pending_wgt, dtype=np.float32)
                b["wgt_en"][0] = 1
            self._pending_wgt = None
            self._run_native(1)
            return

        self._cycle_reference()

    def _shift_weights(self) -> None:
        if not (self.weight_en and self._pending_wgt is not None):
            self._pending_wgt = None
            return
        self._wgt[:, 1:] = self._wgt[:, :-1]
        self._wgt[:, 0] = np.asarray(self._pending_wgt, dtype=np.float32)
        self._pending_wgt = None

    def _shift_activations(self) -> None:
        """Every cycle: columns shift right, column 0 takes the new vector or
        a zero bubble."""
        self._act[:, 1:] = self._act[:, :-1]
        self._col_seq[1:] = self._col_seq[:-1]
        if self._pending_act is not None:
            self._act[:, 0] = np.asarray(self._pending_act, dtype=np.float32)
            self._col_seq[0] = self._next_seq
            self._next_seq += 1
            self.credits = max(0, self.credits - 1)
            self._pending_act = None
        else:
            self._act[:, 0] = 0.0
            self._col_seq[0] = -1

    def _advance_trees(self) -> None:
        """Form this cycle's products and push every column's sum into the
        trees; whatever falls out the far end lands in that column's bank.

        The slot being read is the one about to be overwritten, which is what
        makes the delay exactly pipeline_depth cycles.
        """
        slot = self._tree_head
        done_seq = self._tree_seq[slot]
        sel = (done_seq >= 0) & (self._bank_count < self.output_depth)
        if sel.any():
            reduced = self._reduce_to_dtype(self._tree_vals[slot])
            cols = np.nonzero(sel)[0]
            pos = (self._bank_head[cols] + self._bank_count[cols]) % self.output_depth
            self._bank_vals[cols, pos] = reduced[cols]
            self._bank_seq[cols, pos] = done_seq[cols]
            self._bank_count[cols] += 1

        live = int(np.count_nonzero(self._col_seq >= 0))
        if live:
            products = self._act * self._wgt
            self.metrics["mul_ops"] += live * self.size
            self.metrics["mac_ops"] += live * self.size
            self._tree_vals[slot] = self._reduce_columns(products)
        else:
            self._tree_vals[slot] = 0.0
        self._tree_seq[slot] = self._col_seq
        self._tree_head = (slot + 1) % self.pipeline_depth

    def _drain_output_buffer(self) -> None:
        """A result vector leaves only once every bank holds it -- that is the
        de-skew. One vector per cycle, as in the RTL's single read pointer."""
        if not self._bank_count.all():
            return
        heads = self._bank_seq[self._col_idx, self._bank_head]
        if heads[0] < 0 or not (heads == heads[0]).all():
            return
        row = self._bank_vals[self._col_idx, self._bank_head].tolist()
        self._bank_head += 1
        np.mod(self._bank_head, self.output_depth, out=self._bank_head)
        self._bank_count -= 1
        self._emit(row)

    def _emit(self, row: List[float]) -> None:
        self.output_rows.append(row)
        self._retired += 1
        self.algo_counts["out_rows"] += 1
        self.credits = min(self.max_credits, self.credits + 1)

    def _count_activity(self) -> None:
        """Two NxN comparisons for a number nothing reads unless asked, so it
        only runs when collect_stats is on."""
        self.compute_window_cycles += 1
        if not self.collect_stats:
            return
        active = int(np.count_nonzero((self._act != 0.0) & (self._wgt != 0.0)))
        self.active_pe_sum += active
        self.compute_window_active_pe_sum += active
        self.max_active_pes_in_cycle = max(self.max_active_pes_in_cycle, active)
        if int(np.count_nonzero(self._col_seq >= 0)):
            self.valid_mac_cycles += 1

    # -- stats, with the same names SystolicArrayTPU uses ------------------
    #: Counters the TPU array reports that MEISSA structurally cannot. Named so
    #: a comparison can distinguish "zero" from "no such thing".
    INAPPLICABLE_STATS = ("psum_adds", "psum_shift")

    def _elem_bytes(self) -> int:
        return 1 if self.dtype == DType.INT8 else 2

    def _absorb_metrics(self, m) -> None:
        """Fold one kernel call's metrics into the TPU-shaped counters."""
        eb = self._elem_bytes()
        self.metrics["mul_ops"] += int(m[_native.MM_MUL_OPS])
        self.metrics["mac_ops"] += int(m[_native.MM_MUL_OPS])
        self.metrics["add_ops"] += int(m[_native.MM_ADD_OPS])
        if not self.collect_stats:
            return
        self.active_pe_sum += int(m[_native.MM_ACTIVE_PES])
        self.compute_window_active_pe_sum += int(m[_native.MM_ACTIVE_PES])
        self.valid_mac_cycles += int(m[_native.MM_LIVE_CYCLES])
        self.max_active_pes_in_cycle = max(self.max_active_pes_in_cycle,
                                           int(m[_native.MM_MAX_ACTIVE]))
        self.internal_bytes_valid["act_shift"] += int(m[_native.MM_ACT_SHIFT_NNZ]) * eb
        self.internal_bytes_valid["weight_shift"] += int(m[_native.MM_WGT_SHIFT_NNZ]) * eb
        self.internal_bytes_valid["output"] += int(m[_native.MM_OUT_NNZ]) * eb
        # The reducer narrows FP32 to the storage type, so a finite sum can
        # leave it infinite. Counted the way the TPU counts its casts.
        ovf = int(m[_native.MM_OVERFLOW])
        self.overflow_count += ovf
        self.saturation_count += ovf

    def internal_bytes_total(self) -> int:
        return sum(int(v) for v in self.internal_bytes.values())

    def internal_bytes_valid_total(self) -> int:
        return sum(int(v) for v in self.internal_bytes_valid.values())

    def algo_flops(self) -> int:
        """Useful FLOPs: one multiply and one add per cell per issued row."""
        return 2 * self.algo_counts["act_rows"] * self.size * self.size

    def algo_bytes(self) -> int:
        eb = self._elem_bytes()
        return eb * self.size * (self.algo_counts["act_rows"]
                                 + self.algo_counts["wgt_cols"]
                                 + self.algo_counts["out_rows"])

    def algo_arithmetic_intensity(self) -> float:
        b = self.algo_bytes()
        return (self.algo_flops() / b) if b else 0.0

    def active_lane_count(self) -> int:
        return int(np.count_nonzero((self._act != 0.0) & (self._wgt != 0.0)))

    def active_cell_count(self) -> int:
        both = (self._act != 0.0) & (self._wgt != 0.0)
        return int(np.count_nonzero(both.any(axis=0)))

    # -- egress ------------------------------------------------------------
    def get_buffer(self) -> List[List[float]]:
        """Every result vector produced so far, in order."""
        return self.output_rows

    def pending_outputs(self) -> int:
        return min(len(bank) for bank in self._banks) if self._banks else 0
