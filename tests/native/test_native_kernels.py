"""Correctness tests for the native SIMD kernels.

Two properties matter and are checked separately:

  1. The half-precision cast is bit-identical to numpy's float16 cast, which
     is the semantics the pure-Python simulator has always had.
  2. The native systolic-array tick and the numpy fallback tick produce
     identical state, so building the .so never changes a result.

Everything here is skipped cleanly when the .so has not been built.
"""

import os
import random
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..//..', 'src')))

from base.clock_domain import ClockDomain
from base.core import Core
from base.dtype import DType, cast_scalar, cast_vector
from base.eventq import EventQueue
from base.sim import Sim
from native import kernels as native
from systolic_array.systolic_array_tpu import SystolicArrayTPU

requires_native = pytest.mark.skipif(
    not native.HAVE_NATIVE,
    reason="native kernels not built (run `make native`)")


def numpy_half(values):
    a = np.asarray(values, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        return a.astype(np.float16).astype(np.float64)


def bits(x):
    """Exact bit pattern of a float, so 0.0 vs -0.0 cannot slip through."""
    return struct.pack("<d", float(x)).hex()


def same_bits(got, exp):
    """Exact comparison, treating any two NaNs as equal but not 0.0 vs -0.0."""
    got = np.asarray(got, dtype=np.float64)
    exp = np.asarray(exp, dtype=np.float64)
    both_nan = np.isnan(got) & np.isnan(exp)
    return bool(np.all((got.view(np.uint64) == exp.view(np.uint64)) | both_nan))


# ---------------------------------------------------------------------------
# Half-precision cast
# ---------------------------------------------------------------------------

def half_corpus():
    """Values that between them cover every rounding path into float16."""
    rng = np.random.default_rng(20240607)
    all_halves = np.arange(0, 65536, dtype=np.uint16).view(np.float16).astype(np.float64)
    finite = all_halves[np.isfinite(all_halves)]

    # Midpoints between adjacent halves, and values just either side of them:
    # this is where a naive double->float->half chain misrounds.
    lo = np.arange(0, 31744, dtype=np.uint16).view(np.float16).astype(np.float64)
    hi = np.arange(1, 31745, dtype=np.uint16).view(np.float16).astype(np.float64)
    mid = (lo + hi) / 2.0
    mid = mid[np.isfinite(mid)]

    raw = rng.integers(0, 1 << 64, size=50_000, dtype=np.uint64).view(np.float64)

    return {
        "all finite halves": finite,
        "midpoints": mid,
        "just above midpoints": np.nextafter(mid, np.inf),
        "just below midpoints": np.nextafter(mid, -np.inf),
        "negated midpoints": -mid,
        "subnormals": np.arange(0, 1024, dtype=np.uint16).view(np.float16).astype(np.float64),
        "under smallest subnormal": np.linspace(0.0, 2.0 ** -24, 4000),
        "overflow boundary": np.linspace(65400.0, 65600.0, 8000),
        "negative overflow boundary": np.linspace(-65600.0, -65400.0, 8000),
        "specials": np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 65504.0,
                              65520.0, -65520.0, 1e300, 5e-324, 2049.0]),
        "random raw bits": raw[np.isfinite(raw)],
        "random scaled": rng.uniform(-2, 2, 50_000) * (2.0 ** rng.integers(-30, 18, 50_000)),
    }


@requires_native
@pytest.mark.parametrize("name", sorted(half_corpus().keys()))
def test_native_cast_matches_numpy(name):
    values = half_corpus()[name]
    got = native.cast_array(values, native.CAST_HALF)
    assert same_bits(got, numpy_half(values)), f"native half cast differs on {name}"


@requires_native
def test_native_cast_saturation_and_overflow_counts():
    import ctypes
    values = np.array([1.0, 65504.0, 65505.0, -70000.0, np.inf, np.nan, 0.0])
    stats = (ctypes.c_int64 * 3)()
    native.cast_array(values, native.CAST_HALF, track=True, stats=stats)
    # saturation counts inputs whose magnitude exceeds the largest finite half
    assert stats[0] == int(np.sum(np.abs(values) > 65504.0))
    # overflow counts non-finite results
    assert stats[1] == int(np.sum(~np.isfinite(numpy_half(values))))


@pytest.mark.parametrize("name", sorted(half_corpus().keys()))
def test_cast_scalar_matches_numpy(name):
    """cast_scalar uses struct's 'e' format; it must still match numpy."""
    values = half_corpus()[name]
    got = [cast_scalar(float(v), DType.FP16) for v in values]
    assert same_bits(got, numpy_half(values)), f"cast_scalar differs on {name}"


@pytest.mark.parametrize("n", [1, 8, 31, 32, 33, 256, 1000])
def test_cast_vector_matches_numpy(n):
    """cast_vector switches to the SIMD kernel above a length threshold."""
    rng = np.random.default_rng(n)
    values = (rng.uniform(-2, 2, n) * (2.0 ** rng.integers(-20, 18, n))).tolist()
    assert same_bits(cast_vector(values, DType.FP16), numpy_half(values))


def test_cast_scalar_overflow_saturates_like_numpy():
    assert cast_scalar(70000.0, DType.FP16) == float("inf")
    assert cast_scalar(-70000.0, DType.FP16) == float("-inf")
    assert cast_scalar(65504.0, DType.FP16) == 65504.0


# ---------------------------------------------------------------------------
# Systolic array: native path vs numpy fallback
# ---------------------------------------------------------------------------

def build_array(size, group_size, dtype):
    eq = EventQueue()
    clk = ClockDomain(eq, period=1.0)
    core = Core(eq)
    core.add_clock_domain(clk)
    sim = Sim()
    sim.init(eq, core)
    sa = SystolicArrayTPU(size, boundary_buffer_depth=16, dtype=dtype,
                          group_size=group_size)
    clk.add_clocked(sa)
    return sa


def drive(sa, size, seed, ticks):
    """Run a fixed random workload and return a full state digest per tick."""
    rng = random.Random(seed)

    def val():
        r = rng.random()
        if r < 0.18:
            return 0.0
        if r < 0.28:
            return rng.uniform(-70000, 70000)      # forces fp16 overflow
        if r < 0.38:
            return rng.uniform(-1e-7, 1e-7)        # subnormal region
        return rng.uniform(-40, 40)

    sa.load_weights([[val() for _ in range(size)] for _ in range(size)])
    trace = []
    for t in range(ticks):
        sa.set_control(weight_en=(rng.random() < 0.15),
                       mac_shift=(rng.random() < 0.85),
                       start=(rng.random() < 0.8),
                       stall=(rng.random() < 0.08))
        if rng.random() < 0.7:
            sa.enqueue([val() for _ in range(size)])
        if rng.random() < 0.3:
            sa.enqueue_weights([val() for _ in range(size)])
        if rng.random() < 0.5:
            sa.enqueue_psums([val() for _ in range(size)])
        sa.tick(float(t))

        cells = []
        for g in range(sa.num_groups):
            for j in range(sa.size):
                c = sa.array[g][j]
                cells.append((
                    tuple(bits(v) for v in c.activation_latch),
                    tuple(bits(v) for v in c.weight),
                    bits(c.accumulation),
                    bits(c.mul_reg),
                    c.mul_ops, c.add_ops, c.psum_adds, c.mac_ops,
                ))
        trace.append({
            "cells": cells,
            "metrics": dict(sa.metrics),
            "bytes": dict(sa.internal_bytes),
            "valid": dict(sa.internal_bytes_valid),
            "sat": sa.saturation_count,
            "ovf": sa.overflow_count,
            "active_pe_sum": sa.active_pe_sum,
            "max_active": sa.max_active_pes_in_cycle,
            "valid_cycles": sa.valid_mac_cycles,
            "out": [tuple(bits(v) for v in row) for row in sa.get_buffer()],
        })
    return trace


@requires_native
@pytest.mark.parametrize("size,group_size", [(4, 4), (8, 4), (16, 4), (8, 2), (12, 8)])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_native_matches_fallback(size, group_size, dtype, monkeypatch):
    """The .so and the numpy fallback must agree bit for bit."""
    native_trace = drive(build_array(size, group_size, dtype), size, 7, 40)

    # A SystolicArrayTPU built after this patch allocates its state without the
    # native pointers and ticks through the numpy fallback instead.
    monkeypatch.setattr(native, "HAVE_NATIVE", False)
    fallback_trace = drive(build_array(size, group_size, dtype), size, 7, 40)

    assert len(native_trace) == len(fallback_trace)
    for t, (a, b) in enumerate(zip(native_trace, fallback_trace)):
        assert a == b, f"native and fallback diverge at tick {t}"


@requires_native
def test_isa_is_reported():
    assert native.NATIVE_ISA in ("scalar", "avx2+f16c", "avx512f")
