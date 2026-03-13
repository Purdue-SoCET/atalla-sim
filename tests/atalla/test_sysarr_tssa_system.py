import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..//..", "src")))

from atalla.sysarr_tssa_system import SysArrTSSASystem


def _identity_u16(size: int):
    return [[1 if i == j else 0 for j in range(size)] for i in range(size)]


def _weights_u16(size: int):
    return [[(i * size) + j + 1 for j in range(size)] for i in range(size)]


def test_sysarr_tssa_system_end_to_end():
    size = 32
    system = SysArrTSSASystem(size=size, dtype="fp16", mirror=True)

    act = _identity_u16(size)
    wgt = _weights_u16(size)
    wgt_stream = [[wgt[r][c] for r in range(size)] for c in range(size - 1, -1, -1)]

    system.load_inputs(act, wgt_stream)
    got, mirror = system.run()

    assert mirror is not None
    assert len(mirror) == size
    assert got == mirror


if __name__ == "__main__":
    test_sysarr_tssa_system_end_to_end()
