"""MEISSA with 2:4 structured weight sparsity, Ampere's encoding.

The array keeps its size, its pipeline depth and its timing. Three things
change:

  - a cell holds a SURVIVING weight and a 2-bit index naming which of its
    group's four K positions that weight came from;
  - the activation register carries its group's FOUR values instead of one,
    and every cell muxes the one its index names -- the 4:1 mux per MAC that
    is the whole hardware cost of 2:4;
  - one pass therefore covers 2*size of K instead of size, so a GEMM needs
    half as many passes.

Rows 2g and 2g+1 hold group g's two survivors, so row i reads activation
group i // 2 -- the four values a[4g .. 4g+3] of the injected 2*size vector.

What this does NOT model is the cost of delivering those activations. The
array asks for 2*size of them per cycle where dense asks for size, and the
platform gives each scratchpad pad one VLSU at one issue a cycle
(sysarr_tpu_system.py: "DRAM -> Backend[p] -> pad p -> VLSFrontendBridge ->
VLSU p"). Driven directly, as the tests here drive it, the speedup is a clean
2x; through the platform on one activation pad it cannot be, because the feed
runs out first. tools/run_sparsity_compare.py measures the first case and
says so; the pad split that would fix the second is described there.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np

from systolic_array.systolic_array_meissa import SystolicArrayMEISSA

GROUP = 4          # Ampere's group: 4 consecutive K positions
SURVIVORS = 2      # ... of which 2 survive


class SystolicArrayMEISSASparse24(SystolicArrayMEISSA):
    """MEISSA whose weights are 2:4 packed. See the module docstring.

    Use load_weights_2to4() with a (2*size, size) already-pruned matrix, then
    enqueue() 2*size activations per vector instead of size.
    """

    def __init__(self, size: int, **kwargs) -> None:
        # The native kernel multiplies _act by _wgt elementwise and knows
        # nothing of the mux, so the sparse path stays on the reference one.
        kwargs["use_native"] = False
        super().__init__(size, **kwargs)
        self._act4 = np.zeros((size, size, GROUP), dtype=np.float32)
        self._idx = np.zeros((size, size), dtype=np.int64)

    # -- weights -----------------------------------------------------------

    @property
    def k_per_pass(self) -> int:
        """How much of the reduction dimension one pass covers."""
        return self.size * GROUP // SURVIVORS

    @staticmethod
    def pack_2to4(weights) -> Tuple[np.ndarray, np.ndarray]:
        """Split an already-pruned (2*size, size) matrix into survivors and
        indices, both (size, size).

        A group holding fewer than two non-zeros pads the spare row with a
        zero weight and index 0: the product is zero either way, but the index
        is given a value rather than left undefined, because an X there
        propagates straight through the mux select in RTL simulation.

        A group holding more than two is a weight matrix that was not pruned
        2:4, and is refused rather than silently dropping a real weight.
        """
        w = np.asarray(weights, dtype=np.float32)
        k2, cols = w.shape
        if k2 % GROUP:
            raise ValueError("rows must be a multiple of %d, got %d" % (GROUP, k2))
        rows = k2 // SURVIVORS
        surv = np.zeros((rows, cols), dtype=np.float32)
        idx = np.zeros((rows, cols), dtype=np.int64)
        for j in range(cols):
            for g in range(k2 // GROUP):
                group = w[GROUP * g:GROUP * (g + 1), j]
                nz = np.nonzero(group)[0]
                if len(nz) > SURVIVORS:
                    raise ValueError(
                        "column %d group %d has %d non-zeros; prune 2:4 first"
                        % (j, g, len(nz)))
                for slot in range(SURVIVORS):
                    row = SURVIVORS * g + slot
                    if slot < len(nz):
                        surv[row, j] = group[nz[slot]]
                        idx[row, j] = nz[slot]
        return surv, idx

    def load_weights_2to4(self, weights) -> Tuple[np.ndarray, np.ndarray]:
        """Pack a (2*size, size) pruned matrix and place it. Returns the
        survivors and indices that were loaded."""
        w = np.asarray(weights, dtype=np.float32)
        if w.shape != (self.k_per_pass, self.size):
            raise ValueError("weights must be %d x %d, got %r"
                             % (self.k_per_pass, self.size, w.shape))
        surv, idx = self.pack_2to4(w)
        self.load_weights(surv.tolist())
        self._idx = idx
        return surv, idx

    def metadata_bits(self) -> int:
        """Index bits a packed tile carries: 2 per cell."""
        return self.size * self.size * 2

    # -- activations -------------------------------------------------------

    def enqueue(self, activations: Sequence[float],
                dtype: Optional[object] = None) -> bool:
        """Take the 2*size activations this pass's K span needs."""
        want = self.k_per_pass
        if len(activations) != want:
            raise ValueError("sparse enqueue needs %d activations, got %d"
                             % (want, len(activations)))
        if self._pending_act is not None:
            return False
        self._pending_act = np.asarray(
            self._cast(list(activations), dtype), dtype=np.float32)
        return True

    def _shift_activations(self) -> None:
        """As the base class, but each cell carries its group's four values."""
        self._act4[:, 1:] = self._act4[:, :-1]
        self._col_seq[1:] = self._col_seq[:-1]
        if self._pending_act is not None:
            groups = self._pending_act.reshape(-1, GROUP)
            self._act4[:, 0] = groups[np.arange(self.size) // SURVIVORS]
            self._col_seq[0] = self._next_seq
            self._next_seq += 1
            self.credits = max(0, self.credits - 1)
            self._pending_act = None
        else:
            self._act4[:, 0] = 0.0
            self._col_seq[0] = -1

    def _selected(self) -> np.ndarray:
        """This cycle's activation at every cell, after the 4:1 mux."""
        ii, jj = np.indices((self.size, self.size))
        return self._act4[ii, jj, self._idx]

    # -- the datapath ------------------------------------------------------

    def _advance_trees(self) -> None:
        """The base class's tree step with the product taken through the mux."""
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
            products = self._selected() * self._wgt
            self.metrics["mul_ops"] += live * self.size
            self.metrics["mac_ops"] += live * self.size
            self._tree_vals[slot] = self._reduce_columns(products)
        else:
            self._tree_vals[slot] = 0.0
        self._tree_seq[slot] = self._col_seq
        self._tree_head = (slot + 1) % self.pipeline_depth

    def _count_activity(self) -> None:
        """A PE is active when its MUXED activation and its weight are both
        non-zero -- the dense model's test, read through the mux."""
        self.compute_window_cycles += 1
        if not self.collect_stats:
            return
        active = int(np.count_nonzero(
            (self._selected() != 0.0) & (self._wgt != 0.0)))
        self.active_pe_sum += active
        self.compute_window_active_pe_sum += active
        self.max_active_pes_in_cycle = max(self.max_active_pes_in_cycle, active)
        if int(np.count_nonzero(self._col_seq >= 0)):
            self.valid_mac_cycles += 1

    def _is_drained(self) -> bool:
        if np.any(self._col_seq >= 0) or np.any(self._tree_seq >= 0):
            return False
        return not self._bank_count.any()


def prune_2to4(weights) -> np.ndarray:
    """Prune a dense matrix to 2:4 down K, keeping the two largest of every
    four by magnitude. This is the pruning recipe, not the packing -- the
    array takes its output through load_weights_2to4()."""
    w = np.asarray(weights, dtype=np.float32)
    out = np.zeros_like(w)
    k, cols = w.shape
    for j in range(cols):
        for g in range(k // GROUP):
            group = w[GROUP * g:GROUP * (g + 1), j]
            keep = np.argsort(np.abs(group))[-SURVIVORS:]
            out[GROUP * g + keep, j] = group[keep]
    return out
