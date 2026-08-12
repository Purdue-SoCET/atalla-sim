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
    # BF16: use numpy bfloat16 if available, else fall back to float16.
    try:
        return np.dtype("bfloat16")
    except Exception:
        return np.dtype(np.float16)


# Resolved once at import. The BF16 branch above raises and catches a TypeError
# on every call when numpy lacks bfloat16, which is far too costly to repeat on
# a path that runs O(cells) times per simulated cycle.
_NUMPY_DTYPE = {d: _resolve_numpy_dtype(d) for d in DType}

# Values whose resolved numpy dtype is IEEE binary16 can use struct's 'e'
# format, which is roughly twice as fast as a numpy scalar round-trip and
# rounds identically (round-to-nearest-even). Anything else -- notably a numpy
# build that does provide a real bfloat16 -- keeps the numpy path.
_HALF = np.dtype(np.float16)
_USE_STRUCT_HALF = {d: (_NUMPY_DTYPE[d] == _HALF) for d in DType}

_INF = float("inf")


def numpy_dtype(dtype: DType) -> np.dtype:
    return _NUMPY_DTYPE[dtype]


# Below this length the ctypes call and buffer setup cost more than numpy's
# whole-array cast; measured crossover on this workload is 32 elements.
_NATIVE_VECTOR_MIN = 32


def cast_vector(values: Iterable[float], dtype: DType) -> List[float]:
    # The length test comes first so a one-shot iterable is never partly
    # consumed before falling through to the numpy path below.
    if (_native.HAVE_NATIVE and _USE_STRUCT_HALF[dtype]
            and hasattr(values, "__len__") and len(values) >= _NATIVE_VECTOR_MIN):
        try:
            buf = _array("d", values)
        except (TypeError, ValueError):
            buf = None
        if buf is not None:
            # Cast in place through the SIMD kernel. array('d') exposes a
            # writable buffer, so no copy is needed in either direction.
            _native.cast_buffer_inplace(buf, _native.CAST_HALF)
            return buf.tolist()
    # Saturating to inf is a modelled outcome, not a problem to warn about,
    # and the SIMD path above stays silent -- keep the two consistent.
    with np.errstate(over="ignore", invalid="ignore"):
        arr = np.asarray(list(values), dtype=_NUMPY_DTYPE[dtype])
    if dtype == DType.INT8:
        return [int(x) for x in arr.tolist()]
    return [float(x) for x in arr.tolist()]


def cast_scalar(value: float, dtype: DType) -> float:
    if _USE_STRUCT_HALF[dtype]:
        try:
            return _unpack("e", _pack("e", value))[0]
        except OverflowError:
            # struct raises where numpy saturates to an infinity.
            return -_INF if value < 0 else _INF
        except (TypeError, ValueError):
            pass    # non-float input; fall through to numpy
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.asarray(value, dtype=_NUMPY_DTYPE[dtype]).item()
    if dtype == DType.INT8:
        return int(out)
    return float(out)
