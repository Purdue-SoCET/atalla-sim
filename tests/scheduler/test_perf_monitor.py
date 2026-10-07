"""atalla/perf_monitor.py on the platform: a read-only observer whose
cycle buckets account for every cycle of a run."""
import json

from atalla.atalla_platform import build_atalla_platform

from test_execute import HALT, program
from test_vector import (DRAM_IN, LOAD_ROWS, epilogue, prologue, tile_data, values, vi, vv)


def run(pk, **kw):
    plat = build_atalla_platform(program(*pk, HALT), tile_data(DRAM_IN, LOAD_ROWS, values),
                                 **kw)
    plat.run_until_done(limit=200_000)
    return plat


def test_the_buckets_account_for_every_cycle_and_every_packet():
    pk = prologue() + [(vv("add.vv", 20, 8, 9),), (vv("mul.vv", 21, 20, 10),)] + epilogue([20, 21])
    plat = run(pk)
    r = plat.perf.report()
    assert r["cycles"] == plat.cycle
    assert sum(r["cycle_breakdown"].values()) == r["cycles"]
    assert r["cycle_breakdown"]["issue"] == r["issue"]["packets"] == len(plat.core.issued)
    # the mul waits on the add's result: a hazard on a lane op
    assert r["cycle_breakdown"].get("stall:hazard:vector_lane", 0) > 0
    mix = r["instruction_mix"]
    assert mix["vector_lane"] == 2 and mix["vreg_load"] == LOAD_ROWS and mix["vreg_store"] == 2
    assert mix["sdma_load"] == 1 and mix["sdma_store"] == 1
    # a cold icache: some cycles waiting on fills, which went over DRAM
    assert r["cycle_breakdown"]["frontend:icache"] > 0
    assert r["memory"]["masters"]["icache"]["bytes"] > 0
    assert r["memory"]["masters"]["scpad0"]["bytes"] == LOAD_ROWS * 64


def test_gemm_work_is_counted_from_the_array():
    pk = prologue() + [(vi("lw.vi", 0, 8 + k % 8, 0),) for k in range(32)]
    pk += [(vv("gemm.vv", 20, 9, 0),), (vv("gemm.vv", 21, 10, 0),)]
    plat = run(pk)
    w = plat.perf.report()["work"]
    assert w["array_flops"] == 2 * 2 * 32 * 32          # two vectors through 32x32 MACs
    busy = plat.perf.report()["unit_busy"]
    assert busy["gsau"]["cycles"] >= 32 and busy["array"]["cycles"] > 0   # 32 columns to cross


def test_the_timeline_matches_the_totals(tmp_path):
    pk = prologue() + [(vv("add.vv", 20, 8, 9),)]
    plat = run(pk, perf_timeline=True)
    out = tmp_path / "perf.json"
    plat.perf.to_json(str(out), timeline=True)
    rep = json.loads(out.read_text())
    tl = rep["timeline"]["bucket"]
    assert len(tl) == rep["cycles"] == len(rep["timeline"]["busy"])
    assert tl.count("issue") == rep["issue"]["packets"]
