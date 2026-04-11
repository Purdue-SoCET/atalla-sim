import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..//..", "src")))

from atalla.sysarr_tpu_system import SysArrTPUSystem, build_tpu_platform


def _act_u16(size: int):
    return [[((j * size + i) % 4) + 1 for j in range(size)] for i in range(size)]


def _weights_u16(size: int):
    return [[((i * size + j) % 8) + 1 for j in range(size)] for i in range(size)]


def test_build_tpu_platform_attaches_two_backends_to_shared_dram():
    platform = build_tpu_platform(size=32, dtype="fp16", backend_dram_latency=24)

    assert len(platform.backends) == 2
    assert platform.backend is platform.backends[0]
    assert platform.backends[0] is not platform.backends[1]
    assert platform.spad.backends == platform.backends
    assert platform.spad.backend is platform.backends[0]
    assert all(backend.dram is platform.dram for backend in platform.backends)


def test_sysarr_tpu_system_end_to_end():
    size = 32
    system = SysArrTPUSystem(size=size, dtype="fp16", mirror=True)

    act = _act_u16(size)
    wgt = _weights_u16(size)
    wgt_stream = [[wgt[r][c] for r in range(size)] for c in range(size - 1, -1, -1)]

    system.load_inputs(act, wgt_stream)
    got, mirror, cycles, metrics = system.run()

    assert mirror is not None
    assert len(mirror) == size
    assert got == mirror

    # Metrics from the test configuration.
    print("cycles", cycles)
    print("flops", metrics.flops)
    print("bytes_moved", metrics.bytes_moved)
    print("arithmetic_intensity", metrics.arithmetic_intensity())


if __name__ == "__main__":
    test_build_tpu_platform_attaches_two_backends_to_shared_dram()
    test_sysarr_tpu_system_end_to_end()
