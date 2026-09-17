"""MEISSA: a multiplier grid with per-column adder trees.

Covers the four claims the architecture makes -- the tree shape follows the
array dimension, inputs need no temporal skewing, the output buffer de-skews,
and the array retires one vector per cycle once full.
"""
import numpy as np
import pytest

from base.dtype import cast_vector, normalize_dtype
from systolic_array.systolic_array_meissa import (
    SystolicArrayMEISSA, is_power_of_four, tree_levels)


def _drive(sa, rows, extra=None):
    """Feed every row in, then flush. Returns (outputs, arrival cycles)."""
    arrivals, cyc, sent, seen = [], 0, 0, 0
    limit = len(rows) + sa.flush_cycles() + sa.size + 8 if extra is None else extra
    for _ in range(limit):
        if sent < len(rows) and sa.enqueue(list(rows[sent])):
            sent += 1
        sa.tick(float(cyc))
        cyc += 1
        if len(sa.get_buffer()) > seen:
            seen = len(sa.get_buffer())
            arrivals.append(cyc)
    return sa.get_buffer(), arrivals


# --- the adder tree --------------------------------------------------------

def test_tree_shape_follows_the_array_dimension():
    """16 reduces with 4-input adders alone; 32 cannot and needs a final
    2-input stage, because 32 is not a power of four."""
    assert [is_power_of_four(n) for n in (4, 8, 16, 32, 64)] == \
        [True, False, True, False, True]

    assert tree_levels(16, use_mixed_adder=True) == {"add4": 2, "add2": 0}
    assert tree_levels(32, use_mixed_adder=True) == {"add4": 2, "add2": 1}

    # A pure binary tree is log2(n) deep; four-input levels are log4(n).
    assert tree_levels(32, use_mixed_adder=False) == {"add4": 0, "add2": 5}
    assert tree_levels(16, use_mixed_adder=False) == {"add4": 0, "add2": 4}


def test_pipeline_depth_is_the_multiplier_plus_every_adder_level():
    pure = SystolicArrayMEISSA(size=32, mul_latency=1, add2_latency=1)
    assert pure.pipeline_depth == 1 + 5 * 1

    mixed = SystolicArrayMEISSA(size=32, mul_latency=1, add2_latency=1,
                                add4_latency=3, use_mixed_adder=True)
    assert mixed.pipeline_depth == 1 + 2 * 3 + 1 * 1

    square = SystolicArrayMEISSA(size=16, mul_latency=1, add4_latency=3,
                                 use_mixed_adder=True)
    assert square.pipeline_depth == 1 + 2 * 3, "no 2-input stage for a power of four"


# --- computation -----------------------------------------------------------

def test_a_matrix_product_comes_back_de_skewed():
    n = 8
    rng = np.random.default_rng(0)
    w = rng.integers(1, 5, size=(n, n)).astype(float)
    a = rng.integers(1, 5, size=(n, n)).astype(float)

    sa = SystolicArrayMEISSA(size=n, dtype="bf16")
    sa.load_weights(w.tolist())
    out, _ = _drive(sa, a)

    assert len(out) == n
    assert np.allclose(np.array(out), a @ w), "columns must arrive aligned"


def test_inputs_need_no_temporal_skewing():
    """A whole activation vector is injected on one cycle -- the pipelining in
    the grid and the trees is what removes the need to stagger it."""
    n = 4
    sa = SystolicArrayMEISSA(size=n, dtype="bf16")
    sa.load_weights([[1.0] * n for _ in range(n)])

    assert sa.enqueue([1.0, 2.0, 3.0, 4.0])
    sa.tick(0.0)
    # The entire vector sits in column 0 one cycle later, unstaggered.
    assert sa._act[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert sa._col_seq[0] == 0


def test_one_output_per_cycle_once_the_pipeline_is_full():
    n = 8
    sa = SystolicArrayMEISSA(size=n, dtype="bf16")
    sa.load_weights([[1.0] * n for _ in range(n)])
    rows = [[float(i + 1)] * n for i in range(n)]

    out, arrivals = _drive(sa, rows)

    assert len(out) == n
    assert arrivals[0] == n + sa.pipeline_depth, "first output after the fill"
    assert all(b - a == 1 for a, b in zip(arrivals, arrivals[1:])), \
        "one vector per cycle after that: %s" % arrivals


def test_weights_shift_in_so_the_first_push_lands_in_the_last_column():
    n = 4
    sa = SystolicArrayMEISSA(size=n, dtype="bf16")
    for k in range(n):
        assert sa.enqueue_weights([float(k + 1)] * n)
        sa.set_control(weight_en=True)
        sa.tick(float(k))

    assert sa._wgt[0].tolist() == [4.0, 3.0, 2.0, 1.0]


def test_weights_hold_still_while_weight_en_is_low():
    n = 4
    sa = SystolicArrayMEISSA(size=n, dtype="bf16")
    sa.load_weights([[float(j + 1) for j in range(n)] for _ in range(n)])
    before = sa._wgt.copy()

    sa.set_control(weight_en=False)
    assert sa.enqueue([1.0] * n)
    for c in range(5):
        sa.tick(float(c))

    assert np.array_equal(sa._wgt, before), "activations must not disturb weights"


# --- precision -------------------------------------------------------------

def test_outputs_are_reduced_to_bf16_before_the_buffer():
    """The reducer runs before the output buffer so the banks hold 16-bit
    values, not 32-bit ones."""
    n = 4
    sa = SystolicArrayMEISSA(size=n, dtype="bf16")
    sa.load_weights([[0.1] * n for _ in range(n)])
    out, _ = _drive(sa, [[0.3] * n])

    assert out, "no output"
    bf16 = normalize_dtype("bf16")
    for value in out[0]:
        assert cast_vector([value], bf16)[0] == value, \
            "%r is not representable in bf16" % value
    # and the product really was rounded, not carried at full precision
    assert out[0][0] != 4 * 0.3 * 0.1


def test_the_adder_tree_has_no_psum_input():
    """pipelined_adder_tree.sv drives sum_out from the tree alone; the psum
    port is disconnected, so a non-zero psum would be silently dropped."""
    sa = SystolicArrayMEISSA(size=4, dtype="bf16")
    assert sa.enqueue_psums([0.0] * 4)
    with pytest.raises(ValueError, match="no psum input"):
        sa.enqueue_psums([1.0, 0.0, 0.0, 0.0])


def test_both_tree_shapes_reduce_the_same_exact_values():
    """4-input and 2-input trees round differently, but on values that are
    exact in FP32 they must agree -- which pins the wiring, not the rounding."""
    n = 16
    rng = np.random.default_rng(7)
    w = rng.integers(1, 9, size=(n, n)).astype(float)
    a = rng.integers(1, 9, size=(n, n)).astype(float)

    outs = []
    for mixed in (False, True):
        sa = SystolicArrayMEISSA(size=n, dtype="bf16", use_mixed_adder=mixed)
        sa.load_weights(w.tolist())
        out, _ = _drive(sa, a)
        outs.append(np.array(out))

    assert outs[0].shape == outs[1].shape == (n, n)
    assert np.array_equal(outs[0], outs[1])


# --- flow control ----------------------------------------------------------

def test_credits_stop_the_input_when_the_pipeline_is_full():
    n = 4
    sa = SystolicArrayMEISSA(size=n, dtype="bf16")
    sa.load_weights([[1.0] * n for _ in range(n)])
    assert sa.credits == sa.pipeline_depth + n - 1
    assert sa.ready_in()

    accepted = 0
    for c in range(sa.max_credits + 10):
        if sa.enqueue([1.0] * n):
            accepted += 1
        sa.tick(float(c))
        if not sa.ready_in():
            break

    assert not sa.ready_in(), "credits must run out without drainage"
    assert accepted == sa.max_credits


def test_a_stall_freezes_the_array():
    n = 4
    sa = SystolicArrayMEISSA(size=n, dtype="bf16")
    sa.load_weights([[1.0] * n for _ in range(n)])
    assert sa.enqueue([1.0] * n)
    sa.tick(0.0)
    frozen = sa._act.copy()

    sa.set_control(stall=True)
    for c in range(1, 6):
        sa.tick(float(c))
    assert np.array_equal(sa._act, frozen)

    sa.set_control(stall=False)
    sa.tick(6.0)
    assert not np.array_equal(sa._act, frozen)
