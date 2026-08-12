"""ctypes bindings for the native SIMD kernels.

Loading is best-effort: if ``libatalla_kernels.so`` has not been built, or the
host cannot load it, ``HAVE_NATIVE`` is False and callers use their pure-Python
path. ``ATALLA_NO_NATIVE=1`` in the environment forces that fallback, which is
what the equivalence tests use to compare the two implementations.
"""

import ctypes
import os

import numpy as np

# Cast modes understood by atalla_cast_array / atalla_cast_scalar.
CAST_HALF = 0   # FP16, and BF16 while numpy has no native bfloat16
CAST_INT8 = 1

_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "libatalla_kernels.so")

_c_dbl_p = ctypes.POINTER(ctypes.c_double)
_c_i64_p = ctypes.POINTER(ctypes.c_int64)
_c_u8_p = ctypes.POINTER(ctypes.c_uint8)


class SaState(ctypes.Structure):
    """Mirror of the SaState struct in atalla_kernels.cpp."""

    _fields_ = [
        ("G", ctypes.c_int32),
        ("S", ctypes.c_int32),
        ("GS", ctypes.c_int32),
        ("act", _c_dbl_p),
        ("wgt", _c_dbl_p),
        ("mul", _c_dbl_p),
        ("acc", _c_dbl_p),
    ]


def _bind(lib):
    lib.atalla_isa.restype = ctypes.c_int
    lib.atalla_isa.argtypes = []
    lib.atalla_isa_name.restype = ctypes.c_char_p
    lib.atalla_isa_name.argtypes = []

    lib.atalla_cast_array.restype = None
    lib.atalla_cast_array.argtypes = [
        _c_dbl_p, _c_dbl_p, ctypes.c_int64, ctypes.c_int, ctypes.c_int, _c_i64_p]

    lib.atalla_cast_scalar.restype = ctypes.c_double
    lib.atalla_cast_scalar.argtypes = [
        ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_int32)]

    lib.atalla_sa_tick.restype = None
    lib.atalla_sa_tick.argtypes = [
        ctypes.POINTER(SaState),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,      # start, weight_en, mac_shift
        ctypes.c_int, ctypes.c_int, ctypes.c_int,      # has_dtype, cast_mode, track_sat
        _c_dbl_p, _c_u8_p,                             # psum_top, psum_top_valid
        _c_dbl_p, _c_dbl_p,                            # shift_in, issued
        _c_dbl_p,                                      # issued_out
        _c_i64_p,                                      # metrics
    ]

    lib.atalla_sa_free_scratch.restype = None
    lib.atalla_sa_free_scratch.argtypes = []
    return lib


lib = None
load_error = None
HAVE_NATIVE = False
NATIVE_ISA = "disabled"

if os.environ.get("ATALLA_NO_NATIVE", "") not in ("", "0"):
    load_error = "disabled by ATALLA_NO_NATIVE"
else:
    try:
        lib = _bind(ctypes.CDLL(_LIB_PATH))
        HAVE_NATIVE = True
        NATIVE_ISA = lib.atalla_isa_name().decode()
    except OSError as exc:              # not built, or unloadable on this host
        load_error = str(exc)
        lib = None


def _ptr(arr):
    return arr.ctypes.data_as(_c_dbl_p)


def cast_array(values, mode=CAST_HALF, track=False, stats=None):
    """Cast a float64 array through the SIMD kernel, returning a new array.

    ``stats`` may be a length-3 c_int64 array, which accumulates
    (saturations, overflows, int8-out-of-range). Saturation and overflow are
    only counted when ``track`` is set.
    """
    src = np.ascontiguousarray(values, dtype=np.float64)
    out = np.empty_like(src)
    if stats is None:
        stats = (ctypes.c_int64 * 3)()
    lib.atalla_cast_array(_ptr(src), _ptr(out), src.size, mode,
                          1 if track else 0,
                          ctypes.cast(stats, _c_i64_p))
    return out


def cast_buffer_inplace(buf, mode=CAST_HALF):
    """Cast an array('d') in place through the SIMD kernel.

    The buffer is both source and destination -- each vector lane is stored
    only after it has been loaded, so the aliasing is safe. No stats pointer
    is passed; the kernel skips the counters entirely when it is NULL.
    """
    n = len(buf)
    p = ctypes.cast((ctypes.c_double * n).from_buffer(buf), _c_dbl_p)
    lib.atalla_cast_array(p, p, n, mode, 0, None)


def cast_scalar_native(value, mode=CAST_HALF):
    """Cast one value. Returns (result, out_of_range_flag)."""
    flag = ctypes.c_int32(0)
    out = lib.atalla_cast_scalar(float(value), mode, ctypes.byref(flag))
    return out, bool(flag.value)


class SaArrays:
    """Owns the systolic array's datapath state as flat float64 buffers.

    Layout is lane-major -- ``act[g, lane, j]`` -- so that the column index j
    is contiguous. That is the axis the kernel vectorises over, and it turns
    the per-column systolic shift into one memmove per (group, lane) row.
    """

    __slots__ = ("G", "S", "GS", "act", "wgt", "mul", "acc",
                 "_state", "psum_top", "psum_valid", "shift_in", "issued",
                 "issued_out", "metrics", "_args", "_call")

    def __init__(self, num_groups, size, group_size):
        self.G, self.S, self.GS = int(num_groups), int(size), int(group_size)
        self.act = np.zeros((self.G, self.GS, self.S), dtype=np.float64)
        self.wgt = np.zeros((self.G, self.GS, self.S), dtype=np.float64)
        self.mul = np.zeros((self.G, self.S), dtype=np.float64)
        self.acc = np.zeros((self.G, self.S), dtype=np.float64)

        # Per-tick scratch reused across calls to avoid reallocation.
        self.psum_top = np.zeros(self.S, dtype=np.float64)
        self.psum_valid = np.zeros(self.S, dtype=np.uint8)
        self.shift_in = np.zeros(self.G * self.GS, dtype=np.float64)
        self.issued = np.zeros(self.G * self.GS, dtype=np.float64)
        self.issued_out = np.zeros(self.S, dtype=np.float64)
        self.metrics = np.zeros(6, dtype=np.int64)

        if HAVE_NATIVE:
            self._state = SaState(
                self.G, self.S, self.GS,
                _ptr(self.act), _ptr(self.wgt), _ptr(self.mul), _ptr(self.acc))
            # Every pointer argument is stable for the object's lifetime, so
            # build them once. Rebuilding them per tick costs more than the
            # kernel call itself at small array sizes.
            self._call = lib.atalla_sa_tick
            self._args = (
                ctypes.byref(self._state),
                _ptr(self.psum_top),
                self.psum_valid.ctypes.data_as(_c_u8_p),
                _ptr(self.shift_in),
                _ptr(self.issued),
                _ptr(self.issued_out),
                self.metrics.ctypes.data_as(_c_i64_p),
            )
        else:
            self._state = None
            self._call = None
            self._args = None

    # Metric slot indices, matching the enum in atalla_kernels.cpp.
    M_ACTIVE_PES = 0
    M_PSUM_NNZ = 1
    M_SAT = 2
    M_OVF = 3
    M_SHIFT_NNZ = 4
    M_INT8_RANGE = 5

    def tick(self, start, weight_en, mac_shift, has_dtype, cast_mode, track_sat):
        """Run one simulated cycle in native code. Returns the metrics array."""
        state, psum, valid, shift_in, issued, issued_out, metrics = self._args
        self._call(state, start, weight_en, mac_shift,
                   has_dtype, cast_mode, track_sat,
                   psum, valid, shift_in, issued, issued_out, metrics)
        return self.metrics
