"""The two transpose testbenches: runtime of one rows x 32 tile.

The expected counts are built from the unit's own costs, not read back off a
run: 4 cycles per row pushed (3 through the Clos network, 1 bank write) and
4 per column popped (1 bank read, 3 back through the network). A pop always
drains all 32 columns, so it costs 128 cycles whatever the tile height.
Every bench also checks the transposed data, so these are correctness tests
too.
"""
import pytest

from atalla.transpose_bench import run_spad_bench, run_vrf_bench

PUSH = 4
DRAIN = 32 * 4
#: Spad bench, overlapped: from the first load's issue until the unit takes
#: row 0 (the load's round trip), and from the last column's writeback until
#: its store has committed to the pad's banks.
LOAD_LEAD = 6
STORE_TAIL = 4


@pytest.mark.parametrize("rows", [1, 8, 32])
def test_vrf_bench_costs_the_unit_and_nothing_more(rows):
    """VRF -> transpose -> VRF: pushes go back to back, each column commits
    to the register file as it leaves, so the core adds no cycles."""
    r = run_vrf_bench(rows)
    assert r.cycles == PUSH * rows + DRAIN
    assert len(r.events["col_wb"]) == 32
    assert r.phases["push"] == (0, PUSH * rows - 1)
    assert r.phases["drain"] == (PUSH * rows, PUSH * rows + DRAIN - 1)


@pytest.mark.parametrize("rows", [1, 8, 32])
def test_spad_bench_hides_loads_and_stores_under_the_unit(rows):
    """Loads arrive every 2 cycles and stores leave every 2, both faster than
    the unit's 4 a vector, so with overlap only the first load and the last
    store show: 6 cycles before the transpose and 4 after it."""
    r = run_spad_bench(rows, load_window=4)
    assert r.cycles == LOAD_LEAD + PUSH * rows + DRAIN + STORE_TAIL


@pytest.mark.parametrize("rows", [1, 8, 32])
def test_spad_bench_serialized_adds_every_phase(rows):
    """One phase at a time: 2M + 4 to load, the transpose, 66 to store all
    32 columns (one every 2 cycles, plus the commit)."""
    r = run_spad_bench(rows, overlap=False)
    load = 2 * rows + 4
    store = 2 * 32 + 2
    assert r.cycles == load + PUSH * rows + DRAIN + store
    lo, hi = r.phases["load"]
    assert hi - lo + 1 == load


def test_queueing_every_load_up_front_delays_the_first_push():
    """Packets issue in order. With all 32 loads queued in cycle 0, the first
    push waits behind the ones the core has not issued yet."""
    eager = run_spad_bench(32)
    windowed = run_spad_bench(32, load_window=4)
    assert windowed.cycles == LOAD_LEAD + PUSH * 32 + DRAIN + STORE_TAIL
    assert eager.cycles == windowed.cycles + 2


def test_a_second_pad_for_the_stores_does_not_help_one_tile():
    """The unit, not the VLSU, sets the pace: storing through another pad's
    VLSU leaves the runtime unchanged."""
    same = run_spad_bench(32, load_window=4)
    other = run_spad_bench(32, load_window=4, load_pad=0, store_pad=3)
    assert other.cycles == same.cycles


def test_tile_height_is_checked():
    with pytest.raises(ValueError, match="rows must be in 1..32"):
        run_vrf_bench(0)
    with pytest.raises(ValueError, match="rows must be in 1..32"):
        run_spad_bench(33)
