"""tools/memory_bench.py: the DRAM subsystem report's benchmarks on the model,
on a 64-row slice of the matrix so they run in seconds."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import memory_bench as mb  # noqa: E402
from memory.ramulator import ramulator_available  # noqa: E402


def test_addresses_follow_test_sdma():
    row = mb.sdma_addresses("row", rows=32)
    tile = mb.sdma_addresses("tile", rows=32)
    assert len(row) == len(tile) == 32 * mb.ROW_BYTES // mb.BEAT
    assert row[:3] == [0, 32, 64]
    # a tile row is two beats, then the next row of the same tile
    assert tile[:4] == [mb.TILE_BASE, mb.TILE_BASE + 32,
                        mb.TILE_BASE + mb.ROW_BYTES, mb.TILE_BASE + mb.ROW_BYTES + 32]
    assert sorted(tile) == sorted(a + mb.TILE_BASE for a in row)


def test_the_read_path_streams_a_beat_a_cycle_and_writes_pay_the_gaps():
    """Zero-latency DRAM: reads approach 8 B/cycle; writes lose 2-3 cycles a burst."""
    cycles, beats = mb.axi_random("read", per_master=40)
    assert beats / cycles > 0.98
    cycles, beats = mb.axi_random("write", per_master=40)
    assert 0.6 < beats / cycles < 0.85


@pytest.mark.skipif(not ramulator_available(), reason="needs `make ramulator`")
def test_four_channels_reach_the_wrappers_32_bytes_a_cycle():
    res = mb.run_sdma("table3", ["DDR4 4ch 128b", "DDR4 1ch 128b"], rows=64)
    fast, _m, _t = res[("DDR4 4ch 128b", "row")]
    assert 64 * mb.ROW_BYTES / fast > 31
    slow, m, _t = res[("DDR4 1ch 128b", "row")]
    assert slow > 2.5 * fast and m["stall"] > 50
