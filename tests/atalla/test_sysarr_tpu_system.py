import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..//..", "src")))

from atalla.sysarr_tpu_system import SysArrTPUSystem


def _act_u16(size: int):
    return [[((j * size + i) % 4) + 1 for j in range(size)] for i in range(size)]


def _weights_u16(size: int):
    return [[((i * size + j) % 8) + 1 for j in range(size)] for i in range(size)]


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
    test_sysarr_tpu_system_end_to_end()
