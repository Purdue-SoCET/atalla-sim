import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..//..", "src")))

from base.eventq import EventQueue
from base.clock_domain import ClockDomain
from base.core import Core
from base.sim import Sim

from vector_core.vector_lanes import VectorDatapath

def build_sim():
    eq = EventQueue()
    clk = ClockDomain(eq, period=1.0)
    core = Core(eq)
    core.add_clock_domain(clk)
    sim = Sim()
    sim.init(eq, core)
    return eq, clk, sim


def _run_until_result(sim: Sim, eq: EventQueue, vd: VectorDatapath, max_cycles: int = 128):
    state = {"cycles": 0, "result": None, "time": None}

    def _step(time: float):
        vd.tick(time)
        if vd.result_valid:
            state["result"] = vd.last_result
            state["time"] = time
            return

        state["cycles"] += 1
        if state["cycles"] >= max_cycles:
            return
        eq.schedule(time + 1.0, _step, time + 1.0)

    eq.schedule(0.0, _step, 0.0)
    sim.run()

    if state["result"] is None:
        raise AssertionError("timed out waiting for vector datapath result")
    return state["result"], state["time"]


def _run_until_n_results(
    sim: Sim,
    eq: EventQueue,
    vd: VectorDatapath,
    expected_results: int,
    max_cycles: int = 512,
):
    state = {"cycles": 0, "results": []}

    def _step(time: float):
        vd.tick(time)
        if vd.result_valid:
            state["results"].append(vd.last_result)
        if len(state["results"]) >= expected_results:
            return

        state["cycles"] += 1
        if state["cycles"] >= max_cycles:
            return
        eq.schedule(time + 1.0, _step, time + 1.0)

    eq.schedule(0.0, _step, 0.0)
    sim.run()

    if len(state["results"]) != expected_results:
        raise AssertionError(
            f"timed out waiting for {expected_results} results (got {len(state['results'])})"
        )
    return state["results"]


def test_global_reduction_sum_partial_zero_mode():
    eq, clk, sim = build_sim()
    vd = VectorDatapath(veggie_size=128, lane_count=4, fu_latencies={"alu": 1})
    src0 = [1, 2, 3, 4, 5, 6, 7, 8]
    src1 = [0] * len(src0)

    vd.enqueue(dtype="fp16", 
        src0=src0,
        src1=src1,
        op="add",
        reduce=True,
        reduce_op="sum",
        reduce_out_mode="partial_zero",
    )
    out, _ = _run_until_result(sim, eq, vd)

    assert out["reduction"] == 36
    assert out["vector"] == [36, 0, 0, 0, 0, 0, 0, 0]

def test_global_reduction_min_partial_passthru_mode():
    eq, clk, sim = build_sim()
    vd = VectorDatapath(veggie_size=128, lane_count=4, fu_latencies={"alu": 1})
    src0 = [9, 4, 7, 3, 6, 1, 8, 5]
    src1 = [0] * len(src0)

    vd.enqueue(dtype="fp16", 
        src0=src0,
        src1=src1,
        op="add",
        reduce=True,
        reduce_op="min",
        reduce_out_mode="partial_passthru",
    )
    out, _ = _run_until_result(sim, eq, vd)

    assert out["reduction"] == 1
    assert out["vector"] == [1, 4, 7, 3, 6, 1, 8, 5]


def test_global_reduction_max_broadcast_mode():
    eq, clk, sim = build_sim()
    vd = VectorDatapath(veggie_size=128, lane_count=4, fu_latencies={"alu": 1})
    src0 = [2, 12, 5, 7, 11, 6, 4, 9]
    src1 = [0] * len(src0)
    mask = [True, True, False, True, True, True, True, False]

    vd.enqueue(dtype="fp16", 
        src0=src0,
        src1=src1,
        mask=mask,
        op="add",
        reduce=True,
        reduce_op="max",
        reduce_out_mode="broadcast",
    )
    out, _ = _run_until_result(sim, eq, vd)

    assert out["reduction"] == 12
    assert out["vector"] == [12] * len(src0)


def test_global_reduction_latency_is_n_minus_one_times_alu_latency():
    src0 = [1, 2, 3, 4, 5, 6, 7, 8]
    src1 = [0] * len(src0)
    n = len(src0)
    alu_latency = 2

    eq0, clk0, sim0 = build_sim()
    vd0 = VectorDatapath(veggie_size=128, lane_count=4, fu_latencies={"alu": alu_latency})
    vd0.enqueue(dtype="fp16", src0=src0, src1=src1, op="add", reduce=False)
    _, non_reduce_done_time = _run_until_result(sim0, eq0, vd0)

    eq1, clk1, sim1 = build_sim()
    vd1 = VectorDatapath(veggie_size=128, lane_count=4, fu_latencies={"alu": alu_latency})
    vd1.enqueue(dtype="fp16", 
        src0=src0,
        src1=src1,
        op="add",
        reduce=True,
        reduce_op="sum",
        reduce_out_mode="partial_zero",
    )
    _, reduce_done_time = _run_until_result(sim1, eq1, vd1)

    expected_extra_cycles = (n - 1) * alu_latency
    assert reduce_done_time - non_reduce_done_time == expected_extra_cycles


def test_vector_datapath_integration_mixed_ops_across_lanes():
    eq, clk, sim = build_sim()
    vd = VectorDatapath(
        veggie_size=128,
        lane_count=4,
        issue_width=2,
        fu_latencies={"alu": 1, "sqrt": 3},
    )

    src_add_0 = [1, 2, 3, 4, 5, 6, 7, 8]
    src_add_1 = [8, 7, 6, 5, 4, 3, 2, 1]
    src_sqrt = [1, 4, 9, 16, 25, 36, 49, 64]
    src_mul_0 = [2, 3, 4, 5, 6, 7, 8, 9]
    src_mul_1 = [10] * 8
    mask_mul = [True, False, True, True, False, True, True, False]

    inst0 = vd.enqueue(dtype="fp16", src0=src_add_0, src1=src_add_1, op="add", dst=10, reduce=False)
    inst1 = vd.enqueue(dtype="fp16", src0=src_sqrt, src1=[0] * 8, op="sqrt", dst=11, reduce=False)
    inst2 = vd.enqueue(dtype="fp16", 
        src0=src_mul_0,
        src1=src_mul_1,
        mask=mask_mul,
        op="mul",
        dst=12,
        reduce=False,
    )

    results = _run_until_n_results(sim, eq, vd, expected_results=3)
    got = {r["inst_id"]: r for r in results}

    assert set(got.keys()) == {inst0, inst1, inst2}
    assert got[inst0]["dst"] == 10
    assert got[inst0]["vector"] == [9, 9, 9, 9, 9, 9, 9, 9]
    assert got[inst1]["dst"] == 11
    assert got[inst1]["vector"] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert got[inst2]["dst"] == 12
    assert got[inst2]["vector"] == [20, 0, 40, 50, 0, 70, 80, 0]


def test_vector_datapath_integration_regular_and_reduction_results():
    eq, clk, sim = build_sim()
    vd = VectorDatapath(veggie_size=128, lane_count=4, fu_latencies={"alu": 1})

    inst0 = vd.enqueue(dtype="fp16", 
        src0=[1, 1, 1, 1, 1, 1, 1, 1],
        src1=[2, 2, 2, 2, 2, 2, 2, 2],
        op="add",
        dst=20,
        reduce=False,
    )
    inst1 = vd.enqueue(dtype="fp16", 
        src0=[3, 1, 4, 1, 5, 9, 2, 6],
        src1=[0] * 8,
        op="add",
        dst=21,
        reduce=True,
        reduce_op="max",
        reduce_out_mode="broadcast",
    )

    results = _run_until_n_results(sim, eq, vd, expected_results=2)
    got = {r["inst_id"]: r for r in results}

    assert set(got.keys()) == {inst0, inst1}
    assert got[inst0]["vector"] == [3, 3, 3, 3, 3, 3, 3, 3]
    assert got[inst0]["reduction"] is None
    assert got[inst1]["reduction"] == 9
    assert got[inst1]["vector"] == [9, 9, 9, 9, 9, 9, 9, 9]


if __name__ == "__main__":
    test_global_reduction_sum_partial_zero_mode()
    test_global_reduction_min_partial_passthru_mode()
    test_global_reduction_max_broadcast_mode()
    test_global_reduction_latency_is_n_minus_one_times_alu_latency()
    test_vector_datapath_integration_mixed_ops_across_lanes()
    test_vector_datapath_integration_regular_and_reduction_results()
