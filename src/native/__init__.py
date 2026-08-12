"""Native SIMD kernels for Atalla-Sim.

The shared object is optional. When it is absent every caller transparently
falls back to the pure-Python reference implementation, so the simulator is
always runnable from a clean checkout; build it with ``make native`` for the
accelerated path.
"""

from .kernels import (  # noqa: F401
    HAVE_NATIVE,
    NATIVE_ISA,
    CAST_HALF,
    CAST_INT8,
    SaArrays,
    cast_array,
    cast_buffer_inplace,
    cast_scalar_native,
    lib,
    load_error,
)
