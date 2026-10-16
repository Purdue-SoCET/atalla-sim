"""The arrays' idle fast paths change nothing.

When an array has drained (MEISSA) or settled into a fixed point (TPU), an idle
cycle only adds to counters, so the arrays skip the datapath and add the counts
directly. These tests run the same random traffic -- weight loads, activation
bursts and long idle gaps -- through an array with the fast path and one
without, and compare every piece of state and every counter after every cycle.
"""

import random

import numpy as np
import pytest

from base.queue import SimQueue
from systolic_array.systolic_array_meissa import SystolicArrayMEISSA
from systolic_array.systolic_array_tpu import SystolicArrayTPU


class MeissaNoFastPath(SystolicArrayMEISSA):
    def _is_drained(self):
        return False


class TPUNoFastPath(SystolicArrayTPU):
    @property
    def _steady(self):
        return None

    @_steady.setter
    def _steady(self, value):
        pass


#: Bookkeeping of the fast paths themselves, and handles to native state.
SKIP = {"_drained", "_steady", "_state", "_batch", "_arr", "array"}


def _same(a, b, where):
    if isinstance(a, SimQueue):
        _same(a.items, b.items, where)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b, equal_nan=True), where
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), where
        for k in a:
            _same(a[k], b[k], "%s[%r]" % (where, k))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, "%s[%d]" % (where, i))
    else:
        assert a == b, "%s: %r != %r" % (where, a, b)


def _compare(fast, slow, cycle):
    for name, value in vars(slow).items():
        if name not in SKIP:
            _same(getattr(fast, name), value, "cycle %d: %s" % (cycle, name))
    if isinstance(slow, SystolicArrayTPU):
        for name in ("act", "wgt", "mul", "acc"):
            _same(getattr(fast._arr, name), getattr(slow._arr, name),
                  "cycle %d: _arr.%s" % (cycle, name))


def _traffic(seed, size, cycles):
    """Per cycle: None (idle), ("w", row) or ("a", row)."""
    rng = random.Random(seed)
    plan = []
    while len(plan) < cycles:
        kind = rng.choice(["w", "a", "idle", "idle"])
        n = rng.randint(1, 2 * size) if kind != "idle" else rng.randint(1, 6 * size)
        for _ in range(n):
            if kind == "idle":
                plan.append(None)
            else:
                plan.append((kind, [rng.choice([0.0, rng.uniform(-4, 4)]) for _ in range(size)]))
    return plan[:cycles]


def _drive(sa, step, cycle, dtype):
    """One cycle, as GSAUTPUBridge drives an array."""
    sa.set_control(weight_en=False, mac_shift=False, start=False, stall=False)
    if step is not None:
        kind, row = step
        if kind == "w":
            assert sa.enqueue_weights(row, dtype=dtype)
            sa.set_control(weight_en=True)
        elif sa.enqueue(row, dtype=dtype):
            assert sa.enqueue_psums([0.0] * len(row), dtype=dtype)
            sa.set_control(mac_shift=True, start=True)
    sa.tick(cycle)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("make, slow_cls", [
    (lambda cls, dt: cls(8, dtype=dt, collect_stats=True), MeissaNoFastPath),
    (lambda cls, dt: cls(8, dtype=dt, use_native=False, collect_stats=True), MeissaNoFastPath),
    (lambda cls, dt: cls(8, dtype=dt), TPUNoFastPath),
])
def test_fast_path_matches_the_full_datapath(seed, dtype, make, slow_cls):
    fast_cls = slow_cls.__mro__[1]
    fast, slow = make(fast_cls, dtype), make(slow_cls, dtype)
    for cycle, step in enumerate(_traffic(seed, 8, 600)):
        _drive(fast, step, cycle, dtype)
        _drive(slow, step, cycle, dtype)
        _compare(fast, slow, cycle)


def test_the_fast_paths_are_taken():
    """Otherwise the comparison above proves nothing."""
    m, t = SystolicArrayMEISSA(8, dtype="bf16"), SystolicArrayTPU(8, dtype="bf16")
    for sa in (m, t):
        _drive(sa, ("w", [1.0] * 8), 0, "bf16")
        _drive(sa, ("a", [1.0] * 8), 1, "bf16")
        for cycle in range(2, 200):
            _drive(sa, None, cycle, "bf16")
    assert m._drained
    assert t._steady is not None
