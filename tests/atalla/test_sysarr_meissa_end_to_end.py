"""MEISSA driven through the whole platform.

DRAM -> scratchpad -> VLSU -> vector register file -> GSAU -> MEISSA ->
VRF -> scratchpad -> DRAM, with the GEMM checked against numpy.

This is what the TPU-compatible interface buys: MEISSA needs no bridge of its
own, it goes behind the same GSAUTPUBridge the TPU array uses.
"""
import numpy as np
import pytest

from atalla.sysarr_tpu_system import (
    SysArrTPUSystem, _act_u16, _weights_u16, build_tpu_platform)
from systolic_array.systolic_array_meissa import SystolicArrayMEISSA
from systolic_array.systolic_array_tpu import SystolicArrayTPU


def meissa_weight_stream(w: np.ndarray) -> list:
    """MEISSA loads weights through a shift register, so the first vector
    pushed ends up in the last column: push the columns of w in reverse."""
    size = w.shape[0]
    return [[int(v) for v in w[:, size - 1 - k]] for k in range(size)]


def _run(size):
    system = SysArrTPUSystem(size=size, dtype="fp16", mirror=False,
                             systolic_array="meissa",
                             spad_num_banks=size, spad_bank_size=64)
    act = _act_u16(size)
    w = np.array(_weights_u16(size), dtype=float)
    system.load_inputs(act, meissa_weight_stream(w))
    observed, _expected, cycles, _metrics = system.run(max_cycles=20000)
    # The harness moves fp16 bit patterns, not values.
    decoded = np.array(observed, dtype=np.uint16).view(np.float16).astype(float)
    return system, decoded, np.array(act, dtype=float) @ w, cycles


@pytest.mark.parametrize("size", [4, 8])
def test_meissa_computes_a_tile_gemm_through_the_platform(size):
    system, got, expected, cycles = _run(size)

    assert isinstance(system.sa, SystolicArrayMEISSA)
    assert got.shape == (size, size)
    assert np.allclose(got, expected), "MEISSA GEMM mismatch"
    assert cycles > 0


def test_the_platform_can_build_either_array():
    for name, cls in (("tpu", SystolicArrayTPU), ("meissa", SystolicArrayMEISSA)):
        platform = build_tpu_platform(size=8, dtype="fp16", spad_bank_size=64,
                                      systolic_array=name)
        assert isinstance(platform.sa, cls)
        # the same bridge drives both
        assert platform.sysarr_bridge.sa is platform.sa

    with pytest.raises(ValueError, match="must be 'tpu' or 'meissa'"):
        build_tpu_platform(size=8, spad_bank_size=64, systolic_array="nope")


def test_meissa_asks_the_bridge_for_no_flush_vectors():
    """A skewed array needs zero rows pushed behind the last real one to shift
    its results out. MEISSA does not -- its grid shifts unconditionally and the
    output buffer de-skews -- and pushing them would burn its input credits."""
    platform = build_tpu_platform(size=8, dtype="fp16", spad_bank_size=64,
                                  systolic_array="meissa")
    assert platform.sa.flush_cycles() == 0
    assert platform.sa.warmup_cycles() == 0, "every emitted row is a real result"
    assert platform.sa.drain_cycles() == platform.sa.size + platform.sa.pipeline_depth

    tpu = build_tpu_platform(size=8, dtype="fp16", spad_bank_size=64)
    assert tpu.sa.flush_cycles() > 0, "the TPU array does need flushing"
