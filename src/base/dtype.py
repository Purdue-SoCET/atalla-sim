from array import array as _array
from enum import Enum
from struct import pack as _pack, unpack as _unpack
from typing import Iterable, List, Optional, Union

import numpy as np

from native import kernels as _native


class DType(Enum):
    BF16 = "bf16"
    FP16 = "fp16"
    INT8 = "int8"


_DTYPE_ALIASES = {
    "bf16": DType.BF16,
    "bfloat16": DType.BF16,
    "fp16": DType.FP16,
    "float16": DType.FP16,
    "int8": DType.INT8,
}


def normalize_dtype(value: Optional[Union["DType", str]], default: Optional["DType"] = None) -> Optional["DType"]:
    if value is None:
        return default
    if isinstance(value, DType):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _DTYPE_ALIASES:
            return _DTYPE_ALIASES[key]
    raise ValueError("unsupported dtype: %s" % value)


def _resolve_numpy_dtype(dtype: DType) -> np.dtype:
    if dtype == DType.FP16:
        return np.dtype(np.float16)
    if dtype == DType.INT8:
        return np.dtype(np.int8)
    # BF16: numpy has no bfloat16, so values are held in float32 and rounded
    # by bf16_round. A registered "bfloat16" (ml_dtypes) is deliberately not
    # used: it rounds from double in one step, while the hardware -- and the
    # functional sim -- go through FP32 first.
    return np.dtype(np.float32)


_NUMPY_DTYPE = {d: _resolve_numpy_dtype(d) for d in DType}

_INF = float("inf")


def numpy_dtype(dtype: DType) -> np.dtype:
    """The numpy dtype values of ``dtype`` are stored in. For BF16 this is
    float32, which holds every bfloat16 exactly but does not round to it."""
    return _NUMPY_DTYPE[dtype]


# BF16 rounding, as the functional sim's bf16_round (components/vector_lanes.py)
# does it: round to FP32, then to nearest-even on the top 16 bits of the FP32
# pattern -- add 0x7FFF plus bit 16, clear the low 16 bits. A carry out of the
# mantissa bumps the exponent, so the largest values round up to infinity.
# NaNs are the one case the functional sim gets wrong: a NaN whose low bits
# are all set carries into the sign bit and comes out as -0.0. Here they keep
# their sign and stay NaN (quieted, so the truncated payload is not zero).
_BF16_EXP = 0x7F800000
_BF16_MAN = 0x007FFFFF
_BF16_QUIET = 0x00400000


def bf16_round(values) -> np.ndarray:
    """Round an array to bfloat16, returning it as float32."""
    with np.errstate(over="ignore", invalid="ignore"):
        f = np.asarray(values, dtype=np.float32)
    u = f.view(np.uint32)
    nan = ((u & _BF16_EXP) == _BF16_EXP) & ((u & _BF16_MAN) != 0)
    # NaN lanes are zeroed before the add so it can never wrap.
    num = np.where(nan, np.uint32(0), u)
    r = np.where(nan, u | np.uint32(_BF16_QUIET),
                 num + (np.uint32(0x7FFF) + ((num >> 16) & np.uint32(1))))
    return (r & np.uint32(0xFFFF0000)).astype(np.uint32).view(np.float32)


def _bf16_round_scalar(value: float) -> float:
    try:
        u = _unpack("<I", _pack("<f", value))[0]
    except OverflowError:
        # struct raises where numpy saturates to an infinity.
        return -_INF if value < 0 else _INF
    if (u & _BF16_EXP) == _BF16_EXP and (u & _BF16_MAN):
        u |= _BF16_QUIET
    else:
        u += 0x7FFF + ((u >> 16) & 1)
    return _unpack("<f", _pack("<I", u & 0xFFFF0000))[0]


# Below this length the ctypes call and buffer setup cost more than numpy's
# whole-array cast; measured crossover on this workload is 32 elements.
_NATIVE_VECTOR_MIN = 32

# Casts that take the SIMD kernel for long vectors. INT8 stays on numpy.
_NATIVE_MODE = {DType.FP16: _native.CAST_HALF, DType.BF16: _native.CAST_BF16}


def cast_vector(values: Iterable[float], dtype: DType) -> List[float]:
    # The length test comes first so a one-shot iterable is never partly
    # consumed before falling through to the numpy path below.
    mode = _NATIVE_MODE.get(dtype)
    if (mode is not None and _native.HAVE_NATIVE
            and hasattr(values, "__len__") and len(values) >= _NATIVE_VECTOR_MIN):
        try:
            buf = _array("d", values)
        except (TypeError, ValueError):
            buf = None
        if buf is not None:
            # Cast in place through the SIMD kernel. array('d') exposes a
            # writable buffer, so no copy is needed in either direction.
            _native.cast_buffer_inplace(buf, mode)
            return buf.tolist()
    if dtype == DType.BF16:
        return bf16_round(np.asarray(list(values), dtype=np.float64)).tolist()
    # Saturating to inf is a modelled outcome, not a problem to warn about,
    # and the SIMD path above stays silent -- keep the two consistent.
    with np.errstate(over="ignore", invalid="ignore"):
        arr = np.asarray(list(values), dtype=_NUMPY_DTYPE[dtype])
    if dtype == DType.INT8:
        return [int(x) for x in arr.tolist()]
    return [float(x) for x in arr.tolist()]


def cast_scalar(value: float, dtype: DType) -> float:
    if dtype == DType.FP16:
        # struct's 'e' format is roughly twice as fast as a numpy scalar
        # round-trip and rounds identically (round-to-nearest-even).
        try:
            return _unpack("e", _pack("e", value))[0]
        except OverflowError:
            # struct raises where numpy saturates to an infinity.
            return -_INF if value < 0 else _INF
        except (TypeError, ValueError):
            pass    # non-float input; fall through to numpy
    elif dtype == DType.BF16:
        return _bf16_round_scalar(float(value))
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.asarray(value, dtype=_NUMPY_DTYPE[dtype]).item()
    if dtype == DType.INT8:
        return int(out)
    return float(out)
