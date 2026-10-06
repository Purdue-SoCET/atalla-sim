"""BF16 casts round to bfloat16, not to FP16.

numpy has no bfloat16, and BF16 used to fall back to float16: 10 mantissa bits
instead of 7, and infinity above 65504. The reference now is the functional
sim's bf16_round -- round to FP32, then nearest-even on the top 16 bits -- and
every cast path (cast_scalar, cast_vector on numpy and on the SIMD kernel, the
kernel called directly) must match it bit for bit.
"""

import struct

import numpy as np
import pytest

from base.dtype import DType, bf16_round, cast_scalar, cast_vector
from native import kernels as native
from scheduler import golden

requires_native = pytest.mark.skipif(
    not native.HAVE_NATIVE,
    reason="native kernels not built (run `make native`)")


def f32_bits(values):
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def bf16_corpus():
    rng = np.random.default_rng(20261005)
    # Every bfloat16 bit pattern, as the float32 it is the top half of.
    every = (np.arange(0, 1 << 16, dtype=np.uint32) << 16).view(np.float32)
    every = every[~np.isnan(every)].astype(np.float64)
    # Float32 midpoints between adjacent bfloat16s and their FP32 neighbours.
    mid = ((np.arange(0, 0x7F7F, dtype=np.uint32) << 16) | 0x8000).view(np.float32)
    mid32 = mid.astype(np.float64)
    raw = rng.integers(0, 1 << 32, size=100_000, dtype=np.uint64).astype(np.uint32)
    raw = raw.view(np.float32)
    return {
        "every bf16": every,
        "midpoints": mid32,
        "negated midpoints": -mid32,
        "just above midpoints": np.nextafter(mid, np.float32(np.inf)).astype(np.float64),
        "just below midpoints": np.nextafter(mid, np.float32(-np.inf)).astype(np.float64),
        "random fp32 bits": raw[~np.isnan(raw)].astype(np.float64),
        "random doubles": rng.standard_normal(50_000) * 10.0 ** rng.integers(-40, 40, 50_000),
        "overflow boundary": np.linspace(3.38e38, 3.41e38, 4000),
        "specials": np.array([0.0, -0.0, np.inf, -np.inf, 65504.0, 65520.0,
                              1e39, -1e39, 1e300, 5e-324, 1e-45, 1e-40]),
    }


def golden_bf16_round():
    golden.require(pytest)
    from src.components.vector_lanes import bf16_round as ref
    return ref


def expected(values):
    ref = golden_bf16_round()
    with np.errstate(over="ignore", invalid="ignore"):
        return ref(np.asarray(values, dtype=np.float64))


@pytest.mark.parametrize("name", sorted(bf16_corpus().keys()))
def test_bf16_round_matches_the_functional_sim(name):
    values = bf16_corpus()[name]
    assert np.array_equal(f32_bits(bf16_round(values)), f32_bits(expected(values)))


@pytest.mark.parametrize("name", sorted(bf16_corpus().keys()))
def test_cast_scalar_matches_the_functional_sim(name):
    values = bf16_corpus()[name]
    got = [cast_scalar(float(v), DType.BF16) for v in values]
    assert np.array_equal(f32_bits(got), f32_bits(expected(values)))


@pytest.mark.parametrize("n", [1, 8, 31, 32, 33, 256, 1000])
def test_cast_vector_matches_the_functional_sim(n):
    """Below 32 elements cast_vector uses numpy, above it the SIMD kernel."""
    values = bf16_corpus()["random fp32 bits"][:n].tolist()
    got = cast_vector(values, DType.BF16)
    assert np.array_equal(f32_bits(got), f32_bits(expected(values)))


@requires_native
@pytest.mark.parametrize("name", sorted(bf16_corpus().keys()))
def test_native_cast_matches_the_functional_sim(name):
    values = bf16_corpus()[name]
    got = native.cast_array(values, native.CAST_BF16)
    assert np.array_equal(f32_bits(got), f32_bits(expected(values)))
    one = [native.cast_scalar_native(v, native.CAST_BF16)[0] for v in values[:2000]]
    assert np.array_equal(f32_bits(one), f32_bits(expected(values[:2000])))


def test_bf16_is_not_fp16():
    """The values the FP16 fallback got wrong."""
    # 7 mantissa bits, not 10.
    assert cast_scalar(1.0 / 3.0, DType.BF16) == 0.333984375
    assert cast_scalar(1.0 / 3.0, DType.FP16) == 0.333251953125
    # FP32's exponent range: 65520 is finite, not FP16's infinity.
    assert cast_scalar(65520.0, DType.BF16) == 65536.0
    # Ties go to even.
    assert cast_scalar(1.0 + 2 ** -8, DType.BF16) == 1.0
    assert cast_scalar(1.0 + 3 * 2 ** -8, DType.BF16) == 1.0 + 2 ** -6
    # Rounding up past the largest bfloat16 overflows.
    assert cast_scalar(3.4e38, DType.BF16) == float("inf")
    assert cast_scalar(-3.4e38, DType.BF16) == float("-inf")


def test_bf16_rounds_through_fp32_first():
    """Two roundings, as the hardware does: a double just above a bf16
    midpoint lands exactly on it in FP32, and then ties to even."""
    x = 1.0 + 2 ** -8 + 2 ** -40
    assert cast_scalar(x, DType.BF16) == 1.0
    assert cast_vector([x] * 40, DType.BF16) == [1.0] * 40


def test_bf16_keeps_nans():
    """The functional sim's add would carry a NaN with all-ones low bits into
    the sign bit and return -0.0; a NaN must stay a NaN."""
    nan = struct.unpack("<f", struct.pack("<I", 0x7FFFFFFF))[0]
    assert np.isnan(cast_scalar(nan, DType.BF16))
    assert np.isnan(cast_vector([nan], DType.BF16)[0])
    assert np.isnan(cast_vector([nan] * 40, DType.BF16)).all()
    assert np.isnan(bf16_round(np.array([nan, -nan]))).all()
