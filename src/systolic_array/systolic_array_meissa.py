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

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from base.clocked_object import Clocked
from base.dtype import DType, cast_vector, normalize_dtype

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
        #: Which injected vector currently occupies each column, None = bubble.
        self._col_seq: List[Optional[int]] = [None] * self.size

        # --- adder tree pipelines -------------------------------------------
        # One delay line per column carrying (seq, sum) pairs.
        self._tree: List[List[Optional[tuple]]] = [
            [None] * self.pipeline_depth for _ in range(self.size)
        ]

        # --- output buffer ---------------------------------------------------
        # One bank per column. A result vector is complete when every bank
        # holds it, which is what de-skews the columns.
        self._banks: List[List[tuple]] = [[] for _ in range(self.size)]
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
        self.metrics = {"mul_ops": 0, "add_ops": 0, "mac_ops": 0}
        self.valid_mac_cycles = 0
        self.compute_window_cycles = 0
        self.active_pe_sum = 0

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
    def _reduce(self, terms: np.ndarray) -> np.float32:
        """One column's reduction, in the tree's own shape and order.

        Rounding follows the structure: a 4-input adder rounds once where three
        cascaded 2-input adders round three times, so the two shapes can differ
        in the last bits. Everything stays FP32 until the reducer.
        """
        level = list(np.asarray(terms, dtype=np.float32))
        for _ in range(self.levels["add4"]):
            level = [
                np.float32(np.float32(level[k]) + np.float32(level[k + 1])
                           + np.float32(level[k + 2]) + np.float32(level[k + 3]))
                if k + 3 < len(level) else np.float32(sum(level[k:]))
                for k in range(0, len(level), 4)
            ]
            self.metrics["add_ops"] += len(level)
        while len(level) > 1:
            level = [
                np.float32(level[k] + level[k + 1]) if k + 1 < len(level)
                else np.float32(level[k])
                for k in range(0, len(level), 2)
            ]
            self.metrics["add_ops"] += len(level)
        return np.float32(level[0]) if level else np.float32(0.0)

    def _reducer(self, value: np.float32) -> float:
        """reducer.sv: FP32 down to BF16 before the output buffer, so the banks
        store 16-bit values rather than 32-bit ones."""
        out = float(value)
        if self.dtype is not None:
            out = cast_vector([out], self.dtype)[0]
        return out

    # -- one cycle ---------------------------------------------------------
    def tick(self, time: Optional[Time] = None) -> None:
        cycle = self._consume_tick(time, attr_name="_tick")
        if cycle is None or self.stall:
            return

        self._shift_weights()
        self._shift_activations()
        self._advance_trees()
        self._drain_output_buffer()
        self._count_activity()

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
            self._col_seq[0] = None

    def _advance_trees(self) -> None:
        """Form this cycle's products and push each column's sum into its tree;
        anything falling out the far end lands in that column's bank."""
        for j in range(self.size):
            done = self._tree[j].pop()
            if done is not None:
                seq, value = done
                if len(self._banks[j]) < self.output_depth:
                    self._banks[j].append((seq, self._reducer(value)))
            seq = self._col_seq[j]
            if seq is None:
                self._tree[j].insert(0, None)
                continue
            products = self._act[:, j] * self._wgt[:, j]
            self.metrics["mul_ops"] += self.size
            self.metrics["mac_ops"] += self.size
            self._tree[j].insert(0, (seq, self._reduce(products)))

    def _drain_output_buffer(self) -> None:
        """A result vector leaves only once every bank holds it -- that is the
        de-skew. One vector per cycle, as in the RTL's single read pointer."""
        if any(not bank for bank in self._banks):
            return
        heads = [bank[0][0] for bank in self._banks]
        if len(set(heads)) != 1:
            return
        row = []
        for bank in self._banks:
            _, value = bank.pop(0)
            row.append(value)
        self.output_rows.append(row)
        self._retired += 1
        self.algo_counts["out_rows"] += 1
        self.credits = min(self.max_credits, self.credits + 1)

    def _count_activity(self) -> None:
        active = int(np.count_nonzero((self._act != 0.0) & (self._wgt != 0.0)))
        self.active_pe_sum += active
        self.compute_window_cycles += 1
        if active:
            self.valid_mac_cycles += 1

    # -- egress ------------------------------------------------------------
    def get_buffer(self) -> List[List[float]]:
        """Every result vector produced so far, in order."""
        return self.output_rows

    def pending_outputs(self) -> int:
        return min(len(bank) for bank in self._banks) if self._banks else 0
