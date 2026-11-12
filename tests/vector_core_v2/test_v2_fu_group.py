"""M1: full-width FU groups and the reduction tree (plan section 2).

Timing convention: a token accepted at the end of cycle c is offered to
writeback (sits in `out`) from cycle c + L + 1.
"""
import math
import random

import pytest

from base.dtype import DType, cast_scalar
from vector_core.vector_lanes import GlobalReductionUnit, _op_lut
from vector_core.v2.fu_group import (
    ALUGroup, FUGroup, Token, compare_mask, elementwise, reduce_tree32, reduction_vector)


def tok(i=0, kind="vreg"):
    return Token(kind, dst=i, value=[float(i)] * 32, enable=[True] * 32, seq=i)


def run_until_out(g, accept_cycle=0, max_cycles=50, grant=False):
    """Accept one token in cycle `accept_cycle`; return the first cycle its
    output is offered."""
    for c in range(max_cycles):
        if g.out is not None:
            return c
        g.commit(granted=grant, token=tok() if c == accept_cycle else None)
    raise AssertionError("no output")


@pytest.mark.parametrize("latency", [1, 2, 8, 11, 14])
def test_output_offered_at_c_plus_L_plus_1(latency):
    g = FUGroup("x", latency)
    assert run_until_out(g, accept_cycle=3) == 3 + latency + 1


def test_initiation_interval():
    g = FUGroup("x", 4, ii=2)
    assert g.ready()
    g.commit(token=tok(1))
    assert not g.ready()            # one cycle after an accept, II = 2
    g.commit()
    assert g.ready()


def test_fully_pipelined_one_per_cycle():
    g = FUGroup("x", 3)
    outs = []
    for c in range(12):
        assert g.ready(granted=True)
        if g.out is not None:
            outs.append((c, g.out.dst))
        g.commit(granted=True, token=tok(c) if c < 5 else None)
    assert outs == [(4, 0), (5, 1), (6, 2), (7, 3), (8, 4)]


def test_stall_freezes_without_losing_or_duplicating():
    g = FUGroup("x", 2)
    for c in range(3):
        g.commit(granted=True, token=tok(c))
    # accepted in cycles 0, 1, 2; token 0 is offered from cycle 0 + L + 1 = 3
    assert g.out.dst == 0
    for _ in range(5):                # writeback refuses: frozen
        assert not g.ready(granted=False)
        g.commit(granted=False)
        assert g.out.dst == 0
    seen = []
    for _ in range(6):
        if g.out is not None:
            seen.append(g.out.dst)
        g.commit(granted=True)
    assert seen == [0, 1, 2]
    assert g.idle


def test_reduction_goes_through_the_tree():
    alu = ALUGroup(latency=1, rtree_levels=5)
    first = None
    for c in range(20):
        if alu.rtree.out is not None and first is None:
            first = c
        alu.commit(granted=True, token=tok(7, "reduce") if c == 2 else None,
                   rtree_granted=True)
    assert first == 2 + 1 + 5 + 1        # c + L_alu + 5 + 1
    assert alu.out is None                # never on the ALU's own output


def test_plain_and_reduction_finish_together_on_separate_outputs():
    alu = ALUGroup(latency=1, rtree_levels=5)
    both = []
    for c in range(15):
        if alu.out is not None and alu.rtree.out is not None:
            both.append(c)
        token = tok(1, "reduce") if c == 0 else tok(2) if c == 5 else None
        assert alu.ready_for("reduce" if token and token.kind == "reduce" else "plain",
                             True, True)
        alu.commit(granted=False, token=token, rtree_granted=False) if c >= 7 else \
            alu.commit(granted=True, token=token, rtree_granted=True)
    assert both and both[0] == 7          # no stall, no collision (Q7 revised)


def test_frozen_tree_blocks_reductions_only():
    alu = ALUGroup(latency=1, rtree_levels=1)
    alu.commit(granted=True, token=tok(1, "reduce"))
    alu.commit(granted=True)              # into the tree
    alu.commit(granted=True)              # tree output
    assert alu.rtree.out is not None
    assert not alu.ready_for("reduce", alu_granted=True, rtree_granted=False)
    assert alu.ready_for("plain", alu_granted=True, rtree_granted=False)


# -- numerics: identical element values to the v1 lanes ---------------------------------
OPS = ["add", "sub", "mul", "max", "min", "gt", "lt", "eq", "ne", "exp"]


@pytest.mark.parametrize("op", OPS)
def test_element_values_match_v1(op):
    rng = random.Random(op)
    a = [cast_scalar(rng.uniform(-4, 4), DType.BF16) for _ in range(32)]
    b = [cast_scalar(rng.uniform(-4, 4), DType.BF16) for _ in range(32)]
    mask = [rng.random() < 0.7 for _ in range(32)]
    fn = _op_lut()[op]
    want = [cast_scalar(fn(x, y), DType.BF16) if m else None for x, y, m in zip(a, b, mask)]
    assert elementwise(op, a, b, mask) == want


def test_compare_keeps_old_bits_where_masked():
    a, b = [1.0] * 32, [0.0] * 32
    assert compare_mask("gt", a, b, mask=0x0000FFFF, old=0xABCD0000) == 0xABCDFFFF


@pytest.mark.parametrize("op", ["sum", "min", "max"])
def test_full_mask_reduction_matches_v1_tree(op):
    rng = random.Random(op)
    vals = [cast_scalar(rng.uniform(-2, 2), DType.BF16) for _ in range(32)]
    # v1: each of 16 lanes folds its 2 contiguous elements, then the tree
    pair = {"sum": lambda x, y: x + y, "min": min, "max": max}[op]
    lanes = [pair(0.0 + vals[2 * i], vals[2 * i + 1]) if op == "sum"
             else pair(vals[2 * i], vals[2 * i + 1]) for i in range(16)]
    want = GlobalReductionUnit(32).reduce_tree(lanes, op)
    assert reduce_tree32(vals, [True] * 32, op) == want


def test_full_mask_reduction_matches_golden_lanes_reduce():
    from scheduler.golden import lanes_reduce
    rng = random.Random(5)
    vals = [cast_scalar(rng.uniform(-2, 2), DType.BF16) for _ in range(32)]
    got = cast_scalar(reduce_tree32(vals, [True] * 32, "sum"), DType.BF16)
    assert got == float(lanes_reduce(vals, 0xFFFFFFFF, "sum"))


def test_masked_elements_are_identity():
    vals = [float(i) for i in range(32)]
    mask = [i % 2 == 0 for i in range(32)]
    assert reduce_tree32(vals, mask, "sum") == sum(v for v, m in zip(vals, mask) if m)
    assert reduce_tree32(vals, mask, "max") == 30.0
    assert reduce_tree32(vals, [False] * 32, "max") == -math.inf


def test_reduction_placement():
    vals = [1.0] * 32
    assert reduction_vector(vals, [True] * 32, "sum", "broadcast", 0) == [32.0] * 32
    out = reduction_vector(vals, [True] * 32, "sum", "partial_zero", 3)
    assert out[3] == 32.0 and sum(out) == 32.0
