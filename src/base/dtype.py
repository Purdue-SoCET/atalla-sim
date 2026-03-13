from enum import Enum
from typing import Iterable, List, Optional, Union

import numpy as np


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


def numpy_dtype(dtype: DType) -> np.dtype:
    if dtype == DType.FP16:
        return np.dtype(np.float16)
    if dtype == DType.INT8:
        return np.dtype(np.int8)
    # BF16: use numpy bfloat16 if available, else fall back to float16.
    try:
        return np.dtype("bfloat16")
    except Exception:
        return np.dtype(np.float16)


def cast_vector(values: Iterable[float], dtype: DType) -> List[float]:
    arr = np.asarray(list(values), dtype=numpy_dtype(dtype))
    if dtype == DType.INT8:
        return [int(x) for x in arr.tolist()]
    return [float(x) for x in arr.tolist()]


def cast_scalar(value: float, dtype: DType) -> float:
    out = np.asarray(value, dtype=numpy_dtype(dtype)).item()
    if dtype == DType.INT8:
        return int(out)
    return float(out)
