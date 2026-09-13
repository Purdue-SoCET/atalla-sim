import pytest

from atalla.sysarr_tpu_system import (
    PAD_ACT, PAD_OUT, PAD_PSUM, PAD_WGT, SPAD_NUM_PADS, SysArrTPUSystem,
    build_tpu_platform)


def _act_u16(size: int):
    return [[((j * size + i) % 4) + 1 for j in range(size)] for i in range(size)]


def _weights_u16(size: int):
    return [[((i * size + j) % 8) + 1 for j in range(size)] for i in range(size)]


def test_build_tpu_platform_gives_every_vlsu_its_own_pad():
    """One pad per VLSU: its own banks, frontend and DRAM backend."""
    platform = build_tpu_platform(size=32, dtype="fp16", backend_dram_latency=24,
                                  spad_bank_size=128)
    spad = platform.spad

    n = SPAD_NUM_PADS
    assert len(spad.tiles) == n
    assert len(spad.frontends) == n
    assert len(platform.backends) == n
    assert len(platform.vls_bridges) == n
    assert len(platform.vc.vls_units) == n, "every VLSU must reach memory"

    # pad p is the same index everywhere
    assert [bridge.vls_id for bridge in platform.vls_bridges] == list(range(n))
    assert [bridge.frontend_id for bridge in platform.vls_bridges] == list(range(n))
    assert [fe.tile_id for fe in spad.frontends] == list(range(n))

    # the backends are distinct objects, one attached per pad
    assert len({id(b) for b in platform.backends}) == n
    assert spad.backends == platform.backends
    assert all(backend.dram is platform.dram for backend in platform.backends)

    # the [0] aliases still work
    assert platform.backend is platform.backends[0]
    assert platform.spad.backend is platform.backends[0]
    assert platform.vls_bridge is platform.vls_bridges[0]


def test_pad_count_does_not_change_dram_bandwidth():
    """Scratchpad<->DRAM bandwidth is fixed. More pads means each backend gets a
    smaller share of it, never a bigger total."""
    channels = {}
    for pads in (2, SPAD_NUM_PADS):
        platform = build_tpu_platform(
            size=8, dtype="fp16", vls_count=pads, spad_num_tiles=pads,
            spad_num_banks=8, spad_bank_size=16, backend_dram_latency=4,
            backend_dram_burst_bytes=32,
        )
        assert len(platform.backends) == pads
        # One launch slot shared by every backend is what fixes the aggregate.
        shared = {id(b.shared_burst_channel) for b in platform.backends}
        assert len(shared) == 1, "each pad must not get its own burst channel"
        assert platform.backends[0].shared_burst_channel is not None

        # Give every pad a row to fetch, then tick the platform and count.
        for pad in range(pads):
            platform.dram.write(0x1000 + pad * 0x100, b"\x01\x00" * 8)
            assert platform.backends[pad].driver_to_backend_start_load(
                base_sp_addr=0, base_dram_addr=0x1000 + pad * 0x100,
                rows=1, cols=8) > 0

        per_cycle = []
        for cycle in range(pads * 2):
            before = sum(b.total_dram_bursts_issued for b in platform.backends)
            platform.tick(float(cycle))
            after = sum(b.total_dram_bursts_issued for b in platform.backends)
            per_cycle.append(after - before)

        assert max(per_cycle) <= 1, (
            "aggregate DRAM launches must never exceed one per cycle, got %s"
            % per_cycle)
        # Every pad gets served -- the round-robin does not starve the tail.
        assert all(b.total_dram_bursts_issued == 1 for b in platform.backends), (
            "every backend should have issued exactly once: %s"
            % [b.total_dram_bursts_issued for b in platform.backends])
        channels[pads] = sum(per_cycle)

    # Same bytes requested per pad, same ceiling: doubling the pads doubles the
    # time to drain them, it does not double the bandwidth.
    assert channels[2] == 2 and channels[SPAD_NUM_PADS] == SPAD_NUM_PADS


def test_the_scratchpad_is_two_megabytes_in_four_pads():
    platform = build_tpu_platform(size=32, dtype="fp16", backend_dram_latency=24)
    spad = platform.spad

    assert spad.total_bytes == 2 * 1024 * 1024
    assert spad.tile_bytes == 512 * 1024
    assert spad.num_tiles == SPAD_NUM_PADS
    assert [PAD_ACT, PAD_WGT, PAD_PSUM, PAD_OUT] == [0, 1, 2, 3]


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
    raise SystemExit(pytest.main([__file__]))
