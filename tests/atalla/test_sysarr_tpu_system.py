import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..//..", "src")))

from atalla.sysarr_tpu_system import SysArrTPUSystem, build_tpu_platform


def _act_u16(size: int):
    return [[((j * size + i) % 4) + 1 for j in range(size)] for i in range(size)]


def _weights_u16(size: int):
    return [[((i * size + j) % 8) + 1 for j in range(size)] for i in range(size)]


def test_build_tpu_platform_attaches_two_backends_to_shared_dram():
    platform = build_tpu_platform(size=32, dtype="fp16", vls_count=2, backend_dram_latency=24)

    assert len(platform.backends) == 2
    assert platform.backend is platform.backends[0]
    assert platform.backends[0] is not platform.backends[1]
    assert platform.spad.backends == platform.backends
    assert platform.spad.backend is platform.backends[0]
    assert all(backend.dram is platform.dram for backend in platform.backends)
    assert platform.backends[0].shared_burst_channel is not None
    assert platform.backends[0].shared_burst_channel is platform.backends[1].shared_burst_channel
    assert len(platform.vls_bridges) == 2
    assert platform.vls_bridge is platform.vls_bridges[0]
    assert [bridge.vls_id for bridge in platform.vls_bridges] == [0, 1]
    assert [bridge.frontend_id for bridge in platform.vls_bridges] == [0, 1]


def test_shared_backends_issue_one_dram_burst_per_cycle():
    platform = build_tpu_platform(
        size=8,
        dtype="fp16",
        vls_count=2,
        spad_num_banks=8,
        spad_bank_size=16,
        backend_dram_latency=4,
        backend_dram_q_depth=8,
        backend_dram_burst_bytes=32,
        backend_delay_cycles=1,
    )

    row_bytes = 8 * 2
    platform.dram.write(0x1000, b"\x01\x00" * 8)
    platform.dram.write(0x2000, b"\x02\x00" * 8)
    assert platform.backends[0].driver_to_backend_start_load(base_sp_addr=0, base_dram_addr=0x1000, rows=1, cols=8) > 0
    assert platform.backends[1].driver_to_backend_start_load(base_sp_addr=0, base_dram_addr=0x2000, rows=1, cols=8) > 0

    platform.backends[0].tick(0)
    platform.backends[1].tick(0)
    issued_after_cycle0 = sum(backend.total_dram_bursts_issued for backend in platform.backends)
    assert issued_after_cycle0 == 1

    platform.backends[1].tick(1)
    platform.backends[0].tick(1)
    issued_after_cycle1 = sum(backend.total_dram_bursts_issued for backend in platform.backends)
    assert issued_after_cycle1 == 2


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
    test_shared_backends_issue_one_dram_burst_per_cycle()
    test_sysarr_tpu_system_end_to_end()
