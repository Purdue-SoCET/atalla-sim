"""The 2:4 sparse MEISSA array: packing, the mux, and the halved pass count."""

import numpy as np
import pytest

from systolic_array.systolic_array_meissa import SystolicArrayMEISSA
from systolic_array.systolic_array_meissa_sparse import (
    SystolicArrayMEISSASparse24, prune_2to4)


def _drive(sa, rows):
    """Feed every row in, then flush. Returns (outputs, cycles)."""
    cyc, sent = 0, 0
    limit = len(rows) + sa.flush_cycles() + sa.size + 8
    for _ in range(limit):
        if sent < len(rows) and sa.enqueue(list(rows[sent])):
            sent += 1
        sa.tick(float(cyc))
        cyc += 1
    return sa.get_buffer(), cyc


# -- packing ---------------------------------------------------------------

def test_packing_keeps_two_of_every_four_with_their_positions():
    n = 4
    w = np.zeros((2 * n, n), dtype=np.float32)
    # column 0, group 0: survivors at K positions 1 and 3
    w[1, 0], w[3, 0] = 5.0, 7.0
    surv, idx = SystolicArrayMEISSASparse24.pack_2to4(w)

    assert surv.shape == (n, n) and idx.shape == (n, n)
    assert surv[0, 0] == 5.0 and idx[0, 0] == 1
    assert surv[1, 0] == 7.0 and idx[1, 0] == 3


def test_a_group_with_one_survivor_pads_with_a_defined_index():
    """An index is written even where the weight is zero: an undefined select
    propagates X through the mux in RTL simulation."""
    n = 4
    w = np.zeros((2 * n, n), dtype=np.float32)
    w[2, 0] = 9.0                      # group 0 holds a single non-zero
    surv, idx = SystolicArrayMEISSASparse24.pack_2to4(w)

    assert surv[0, 0] == 9.0 and idx[0, 0] == 2
    assert surv[1, 0] == 0.0 and idx[1, 0] == 0


def test_a_matrix_that_was_not_pruned_2to4_is_refused():
    n = 4
    w = np.ones((2 * n, n), dtype=np.float32)      # four per group
    with pytest.raises(ValueError, match="prune 2:4 first"):
        SystolicArrayMEISSASparse24.pack_2to4(w)


def test_a_tile_carries_two_index_bits_per_cell():
    sa = SystolicArrayMEISSASparse24(size=32, dtype="bf16")
    assert sa.metadata_bits() == 32 * 32 * 2
    assert sa.metadata_bits() // 8 == 256           # bytes, exactly


# -- the mux ---------------------------------------------------------------

def test_each_cell_reads_the_activation_its_index_names():
    n = 4
    w = np.zeros((2 * n, n), dtype=np.float32)
    for j in range(n):
        w[1, j] = 1.0                  # every column's first survivor is K=1
    sa = SystolicArrayMEISSASparse24(size=n, dtype="bf16")
    sa.load_weights_2to4(w)

    acts = [float(k) for k in range(2 * n)]
    assert sa.enqueue(acts)
    sa.tick(0.0)

    # row 0 belongs to group 0, and its index says position 1 -> a[1]
    assert sa._act4[0, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert sa._selected()[0, 0] == 1.0


def test_enqueue_wants_the_whole_k_span():
    n = 8
    sa = SystolicArrayMEISSASparse24(size=n, dtype="bf16")
    assert sa.k_per_pass == 2 * n
    with pytest.raises(ValueError, match="needs 16 activations"):
        sa.enqueue([1.0] * n)


# -- the product -----------------------------------------------------------

@pytest.mark.parametrize("n", [4, 8, 16])
def test_a_packed_pass_matches_the_dense_product_of_the_pruned_weights(n):
    rng = np.random.default_rng(3)
    w = prune_2to4(rng.uniform(-2, 2, (2 * n, n)).astype(np.float32))
    a = rng.uniform(-2, 2, (n, 2 * n)).astype(np.float32)

    sa = SystolicArrayMEISSASparse24(size=n, dtype="bf16")
    sa.load_weights_2to4(w)
    out, _ = _drive(sa, a)

    got = np.array(out[:n], dtype=np.float32)
    want = a @ w
    assert np.allclose(got, want, rtol=0.05, atol=0.5 * np.abs(want).mean())


def test_one_packed_pass_replaces_two_dense_ones():
    """The point of 2:4: the same K span, half the passes, same answer."""
    n = 8
    rng = np.random.default_rng(5)
    w = prune_2to4(rng.uniform(-2, 2, (2 * n, n)).astype(np.float32))
    a = rng.uniform(-2, 2, (n, 2 * n)).astype(np.float32)

    sparse = SystolicArrayMEISSASparse24(size=n, dtype="bf16")
    sparse.load_weights_2to4(w)
    got_sparse, cyc_sparse = _drive(sparse, a)

    dense_total = np.zeros((n, n), dtype=np.float32)
    cyc_dense = 0
    for k0 in (0, n):
        d = SystolicArrayMEISSA(size=n, dtype="bf16")
        d.load_weights(w[k0:k0 + n, :].tolist())
        out, c = _drive(d, a[:, k0:k0 + n])
        dense_total += np.array(out[:n], dtype=np.float32)
        cyc_dense += c

    assert np.allclose(np.array(got_sparse[:n]), dense_total,
                       rtol=0.05, atol=0.5 * np.abs(dense_total).mean())
    assert cyc_sparse < cyc_dense


def test_timing_is_unchanged_so_the_saving_is_the_pass_count():
    """A packed pass costs exactly what a dense pass costs -- the mux is
    combinational, so 2:4 buys its speedup by running fewer passes, not by
    running them faster."""
    n = 8
    rng = np.random.default_rng(11)
    w = prune_2to4(rng.uniform(-2, 2, (2 * n, n)).astype(np.float32))
    a = rng.uniform(-2, 2, (n, 2 * n)).astype(np.float32)

    sparse = SystolicArrayMEISSASparse24(size=n, dtype="bf16")
    sparse.load_weights_2to4(w)
    _, cyc_sparse = _drive(sparse, a)

    dense = SystolicArrayMEISSA(size=n, dtype="bf16")
    dense.load_weights(w[:n, :].tolist())
    _, cyc_dense = _drive(dense, a[:, :n])

    assert sparse.pipeline_depth == dense.pipeline_depth
    assert cyc_sparse == cyc_dense


def test_utilisation_doubles_on_the_same_pruned_weights():
    n = 16
    rng = np.random.default_rng(13)
    w = prune_2to4(rng.uniform(-2, 2, (2 * n, n)).astype(np.float32))
    a = rng.uniform(-2, 2, (4 * n, 2 * n)).astype(np.float32)

    sparse = SystolicArrayMEISSASparse24(size=n, dtype="bf16", collect_stats=True)
    sparse.load_weights_2to4(w)
    _drive(sparse, a)
    u_sparse = sparse.active_pe_sum / (sparse.compute_window_cycles * n * n)

    dense = SystolicArrayMEISSA(size=n, dtype="bf16", collect_stats=True)
    dense.load_weights(w[:n, :].tolist())
    _drive(dense, a[:, :n])
    u_dense = dense.active_pe_sum / (dense.compute_window_cycles * n * n)

    assert u_sparse > 1.5 * u_dense
