"""The two transpose testbenches: runtime of one rows x 32 tile.

The expected counts are built from the unit's own costs, which match the RTL
cycle for cycle (test_transpose_rtl_trace.py), not read back off a run:
5 cycles per row pushed (accept, 3 in the Clos network, 1 until the bank's
write done) and 6 per column popped (POPPING, 1 until read done, 3 in the
network, DONE). A pop always drains all 32 columns, so it costs 192 cycles
whatever the tile height.
Every bench also checks the transposed data, so these are correctness tests
too.
"""
import pytest

from atalla.transpose_bench import run_spad_bench, run_vrf_bench
from vector_core.transpose import (
    CLOS_LATENCY, SRAM_READ_LATENCY, SRAM_WRITE_LATENCY, bank_done_delay)

#: A row's cost in the transpose unit (5), and a full tile's 32 columns (192);
#: see push_cycles / column_cycles in tests/vector_core/test_transpose_unit.py.
PUSH = 1 + CLOS_LATENCY + bank_done_delay(SRAM_WRITE_LATENCY)
DRAIN = 32 * (1 + bank_done_delay(SRAM_READ_LATENCY) + CLOS_LATENCY + 1)
#: Spad bench, overlapped. Lead: the first load's 6-cycle scratchpad read,
#: a cycle for the VLSU to write the row back, a cycle to issue the push.
#: Tail: the last column's store is issued the cycle after its writeback and
#: reaches the banks 2 cycles after the pad accepts it.
LOAD_LEAD = 8
STORE_TAIL = 3
#: A pad reads one row every 2 cycles and writes one every cycle (sram_bank
#: read 2, write 1).
READ_INTERVAL = 2
WRITE_INTERVAL = 1


@pytest.mark.parametrize("rows", [1, 8, 32])
def test_vrf_bench_costs_the_unit_and_nothing_more(rows):
    """VRF -> transpose -> VRF: pushes go back to back, the pop is taken the
    cycle the last push finishes, and each column commits to the register
    file as it leaves, so the core adds no cycles."""
    r = run_vrf_bench(rows)
    assert r.cycles == PUSH * rows + DRAIN
    assert len(r.events["col_wb"]) == 32
    assert r.phases["push"] == (0, PUSH * rows - 1)
    assert r.phases["drain"] == (PUSH * rows, PUSH * rows + DRAIN - 1)


@pytest.mark.parametrize("rows", [1, 8, 32])
def test_spad_bench_hides_loads_and_stores_under_the_unit(rows):
    """Loads arrive every 2 cycles and stores leave every cycle -- one row per
    sram_bank access -- both faster than the unit's 5 a row and 6 a column,
    so with overlap only the first load and the last store show."""
    r = run_spad_bench(rows, load_window=4)
    assert r.cycles == LOAD_LEAD + PUSH * rows + DRAIN + STORE_TAIL


@pytest.mark.parametrize("rows", [1, 8, 32])
def test_spad_bench_serialized_adds_every_phase(rows):
    """One phase at a time: a row loaded every 2 cycles, the last one's lead
    included (2M + 6); the transpose; then 32 column stores at one a cycle,
    the last one's tail included (31 + 3)."""
    r = run_spad_bench(rows, overlap=False)
    load = READ_INTERVAL * (rows - 1) + LOAD_LEAD
    store = WRITE_INTERVAL * 31 + STORE_TAIL
    assert r.cycles == load + PUSH * rows + DRAIN + store
    lo, hi = r.phases["load"]
    assert hi - lo + 1 == load


def test_queueing_every_load_up_front_costs_nothing():
    """The pad takes a request every cycle into a 32-deep queue, so loads
    queued all at once drain out of the VLSU at once and never hold up the
    first push behind them."""
    eager = run_spad_bench(32)
    windowed = run_spad_bench(32, load_window=4)
    assert eager.cycles == windowed.cycles == \
        LOAD_LEAD + PUSH * 32 + DRAIN + STORE_TAIL


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
