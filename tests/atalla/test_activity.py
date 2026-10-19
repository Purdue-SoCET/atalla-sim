"""The activity probe (src/atalla/activity.py) and its viewer (tools/activity_view.py).

The probe only reads the platform, so a run with it is the same run; its
counts agree with the performance monitor's where both measure the same
thing; its intervals are a faithful, compact record; and the viewer draws
every block it recorded.
"""
import gzip
import json
import re
import sys
from pathlib import Path

import pytest

from atalla.activity import ACTIVE, IDLE, STALLED, ActivityProbe
from atalla.atalla_platform import build_atalla_platform
from scheduler.assembler import assemble
from scheduler.core import load_program_text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import activity_view  # noqa: E402

PROGRAMS = ROOT / "tests" / "equivalence" / "programs"


def frozen(name):
    return load_program_text(gzip.decompress((PROGRAMS / (name + ".in.gz")).read_bytes()).decode())


@pytest.fixture(scope="module")
def gemm():
    """The gemm kernel, run once with the probe."""
    instr, data = frozen("gemm")
    plat = build_atalla_platform(instr, data, activity=True)
    plat.run_until_done()
    return plat


def test_the_probe_changes_nothing(gemm):
    instr, data = frozen("gemm")
    plain = build_atalla_platform(instr, data)
    plain.run_until_done()
    assert plain.cycle == gemm.cycle
    assert plain.perf.report() == gemm.perf.report()
    assert [plain.vc.read_vreg(v) for v in range(64)] == [gemm.vc.read_vreg(v) for v in range(64)]
    assert plain.dram._blocks == gemm.dram._blocks


def test_counts_agree_with_the_performance_monitor(gemm):
    """Where the two measure the same thing. The monitor counts a unit busy
    whether it is moving work or held up, so active + stalled."""
    a, busy = gemm.activity, gemm.perf.busy
    same = {"gsau": "gsau", "array": "array", "channel": "dram_channel",
            "ex3": "ex3", "ex4": "ex4", "lsu": "lsu", "dcache": "dcache",
            **{"vlsu%d" % i: "vlsu%d" % i for i in range(4)},
            **{"sdma%d" % i: "sdma%d" % i for i in range(4)}}
    for bid, unit in same.items():
        k = a.counts(bid)
        assert k["active"] + k["stalled"] == busy.get(unit, 0), bid
    ex2 = sum(a.counts(b)["active"] for b in ("ex2_div", "ex2_cvt"))
    assert ex2 == busy.get("ex2", 0), "EX2's two kinds of op together are EX2"
    assert a.counts("gsau")["active"] > 0 and a.counts("array")["active"] > 0


def test_intervals_are_sorted_merged_and_inside_the_run(gemm):
    a = gemm.activity
    assert a.cycles == gemm.perf.cycles
    for bid, runs in a.intervals.items():
        for first, last, state in runs:
            assert 0 <= first <= last < a.cycles and state in (ACTIVE, STALLED), bid
        for (f0, l0, s0), (f1, l1, s1) in zip(runs, runs[1:]):
            assert l0 < f1, "%s: overlapping or unsorted" % bid
            assert not (s0 == s1 and l0 + 1 == f1), "%s: unmerged runs" % bid
    bucket_cycles = sum(last - first + 1 for first, last, _ in a.buckets)
    assert bucket_cycles == a.cycles


def test_a_scalar_program_lights_no_vector_array_or_dma_block():
    text = "\n".join(["main:",
                      "addi_s x5, x0, 3", "nop", "nop", "nop",
                      "loop:",
                      "addi_s x5, x5, -1", "nop", "nop", "nop",
                      "blt_s x0, x5, loop", "nop", "nop", "nop",
                      "jalr x0, x1, 0", "nop", "nop", "nop"])
    plat = build_atalla_platform(assemble(text).instr, {}, activity=True)
    plat.run_until_done()
    a = plat.activity
    quiet = [b.id for b in a.blocks
             if b.panel in ("Vector core", "Systolic array") or b.id.startswith(("spad", "sdma"))]
    assert quiet
    for bid in quiet:
        assert a.intervals[bid] == [], bid
    assert a.counts("ex1_alu")["active"] > 0 and a.counts("ex1_br")["active"] > 0
    assert sum(a.counts("d2_s%d" % i)["active"] for i in range(4)) > 0


def test_the_viewer_draws_every_block(gemm, tmp_path):
    trace = gemm.activity.trace({"program": "gemm"})
    page = activity_view.render(trace)
    embedded = re.search(r'<script id="trace" type="application/json">(.*?)</script>', page, re.S)
    assert json.loads(embedded.group(1)) == json.loads(json.dumps(trace))
    assert "<title>gemm activity</title>" in page
    assert "Percentages: the share of the run" in page
    path = tmp_path / "t.json"
    gemm.activity.to_json(str(path), {"program": "gemm"})
    assert {b["id"] for b in json.loads(path.read_text())["blocks"]} == {b.id for b in gemm.activity.blocks}


def test_states_are_named_in_order():
    assert (IDLE, ACTIVE, STALLED) == (0, 1, 2)
    assert ActivityProbe.__doc__
