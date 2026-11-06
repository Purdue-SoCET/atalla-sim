"""The first-divergence report (src/atalla/lockstep.py)."""
import sys
from pathlib import Path

import pytest

from atalla import lockstep
from atalla.atalla_platform import build_atalla_platform
from scheduler import golden

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scheduler"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_execute import HALT, program  # noqa: E402
from test_vector import DRAM_IN, LOAD_ROWS, tile_data, values  # noqa: E402
from test_check import load  # noqa: E402

pytestmark = [pytest.mark.golden,
              pytest.mark.skipif(not golden.HAVE_GOLDEN, reason="no functional sim")]


def run(prog):
    data = tile_data(DRAM_IN, LOAD_ROWS, values)
    plat = build_atalla_platform(prog, data, strict=False)
    rec = lockstep.attach(plat)
    plat.run_until_done(limit=200_000)
    trace = []
    golden.run_golden_state(prog, data, reductions="lanes", trace=trace)
    return plat, rec, trace


def test_a_correctly_ordered_load_matches_write_for_write():
    plat, rec, trace = run(program(*load(addr_reg=2), HALT))
    assert lockstep.first_divergence(rec, trace) is None
    assert any(k == "v" for k, *_ in trace)


def test_a_load_ahead_of_its_dma_is_the_first_divergence():
    prog = program(*load(addr_reg=5), HALT)
    plat, rec, trace = run(prog)
    d = lockstep.first_divergence(rec, trace)
    assert d is not None and d.kind == "v" and d.index == 0
    text = lockstep.describe(d, prog)
    assert "vreg.ld" in text and "functional sim" in text


def test_a_lost_write_is_reported_as_never_made():
    rec = lockstep.WriteRecorder()
    rec.writes[("x", 5)] = [(10, 3)]
    d = lockstep.first_divergence(rec, [("x", 5, 3, 0x28), ("x", 6, 4, 0x28)])
    assert d.kind == "x" and d.reg == 6 and d.cycle is None and d.pc == 0x28
    assert "never made write #1 to x6" in lockstep.describe(d, {})
