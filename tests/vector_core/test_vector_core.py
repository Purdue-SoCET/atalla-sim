import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..//..", "src")))

from base.eventq import EventQueue
from base.clock_domain import ClockDomain
from base.core import Core
from base.sim import Sim

from vector_core.vector_core import VectorCore


def build_sim():
    eq = EventQueue()
    clk = ClockDomain(eq, period=1.0)
    core = Core(eq)
    core.add_clock_domain(clk)
    sim = Sim()
    sim.init(eq, core)
    return eq, clk, sim


def _run_core_until(
    sim: Sim,
    eq: EventQueue,
    vc: VectorCore,
    done_cb,
    max_cycles: int = 128,
):
    state = {"cycles": 0}

    def _step(time: float):
        vc.tick()
        if done_cb():
            return
        state["cycles"] += 1
        if state["cycles"] >= max_cycles:
            return
        eq.schedule(time + 1.0, _step, time + 1.0)

    eq.schedule(0.0, _step, 0.0)
    sim.run()
    if not done_cb():
        raise AssertionError("timed out waiting for vector core completion")


def _run_core_with_scratchpad(
    sim: Sim,
    eq: EventQueue,
    vc: VectorCore,
    memory: dict,
    done_cb,
    max_cycles: int = 128,
):
    state = {"cycles": 0, "pending_rsp": []}

    def _step(time: float):
        due = [e for e in state["pending_rsp"] if e["time"] <= time]
        state["pending_rsp"] = [e for e in state["pending_rsp"] if e["time"] > time]
        for e in due:
            assert vc.push_scratchpad_response(0, e["rsp"])

        vc.tick()

        while True:
            req = vc.pop_scratchpad_request(0)
            if req is None:
                break
            addr = req.get("addr")
            if req["kind"] == "store":
                memory[addr] = list(req["data"])
            else:
                state["pending_rsp"].append(
                    {
                        "time": time + 1.0,
                        "rsp": {
                            "addr": addr,
                            "data": list(memory.get(addr, [0] * vc.vector_len)),
                        },
                    }
                )

        if done_cb():
            return
        state["cycles"] += 1
        if state["cycles"] >= max_cycles:
            return
        eq.schedule(time + 1.0, _step, time + 1.0)

    eq.schedule(0.0, _step, 0.0)
    sim.run()
    if not done_cb():
        raise AssertionError("timed out waiting for vector core + scratchpad completion")


def _run_core_with_systolic(
    sim: Sim,
    eq: EventQueue,
    vc: VectorCore,
    done_cb,
    transform_cb,
    rsp_latency: int = 2,
    max_cycles: int = 256,
):
    state = {"cycles": 0, "pending_rsp": []}

    def _step(time: float):
        due = [e for e in state["pending_rsp"] if e["time"] <= time]
        state["pending_rsp"] = [e for e in state["pending_rsp"] if e["time"] > time]
        for e in due:
            assert vc.push_systolic_response(e["rsp"])

        vc.tick()

        while True:
            req = vc.pop_systolic_request()
            if req is None:
                break
            if req.get("expect_output", True):
                state["pending_rsp"].append(
                    {
                        "time": time + float(rsp_latency),
                        "rsp": {
                            "vdata": list(transform_cb(req)),
                            "meta": {"echo_is_weight": bool(req.get("is_weight", False))},
                        },
                    }
                )

        if done_cb():
            return
        state["cycles"] += 1
        if state["cycles"] >= max_cycles:
            return
        eq.schedule(time + 1.0, _step, time + 1.0)

    eq.schedule(0.0, _step, 0.0)
    sim.run()
    if not done_cb():
        raise AssertionError("timed out waiting for vector core + systolic completion")


def test_vector_core_sim_compute_writeback():
    eq, clk, sim = build_sim()
    vc = VectorCore(veggie_size=128, lane_count=4, fu_latencies={"alu": 1})
    vc.load_vreg(1, [1, 2, 3, 4, 5, 6, 7, 8])
    vc.load_vreg(2, [8, 7, 6, 5, 4, 3, 2, 1])

    assert vc.enqueue_compute(op="add", dst=3, src0=1, src1=2)

    _run_core_until(
        sim,
        eq,
        vc,
        done_cb=lambda: (
            vc.wb_valid
            and vc.last_wb is not None
            and vc.last_wb.get("source") == "datapath"
            and vc.last_wb.get("dst") == 3
        ),
    )

    assert vc.dump_vreg(3) == [9, 9, 9, 9, 9, 9, 9, 9]
    assert vc.last_wb["meta"]["op"] == "add"


def test_vector_core_sim_vlsu_store_then_load_round_trip():
    eq, clk, sim = build_sim()
    vc = VectorCore(veggie_size=128, lane_count=4, vls_count=1, fu_latencies={"alu": 1})
    src_vec = [11, 22, 33, 44, 55, 66, 77, 88]
    vc.load_vreg(6, src_vec)
    memory = {}

    assert vc.enqueue_memory({"kind": "store", "vls": 0, "src": 6, "addr": 0x1000})
    assert vc.enqueue_memory({"kind": "load", "vls": 0, "dst": 7, "addr": 0x1000})

    _run_core_with_scratchpad(
        sim,
        eq,
        vc,
        memory,
        done_cb=lambda: (
            vc.wb_valid
            and vc.last_wb is not None
            and vc.last_wb.get("source") == "vlsu"
            and vc.last_wb.get("dst") == 7
        ),
    )

    assert memory[0x1000] == src_vec
    assert vc.dump_vreg(7) == src_vec


def test_vector_core_sim_gsau_round_trip_with_rd_queue():
    eq, clk, sim = build_sim()
    vc = VectorCore(veggie_size=128, lane_count=4, fu_latencies={"alu": 1})
    src0 = [1, 2, 3, 4, 5, 6, 7, 8]
    src1 = [9, 8, 7, 6, 5, 4, 3, 2]
    vc.load_vreg(20, src0)
    vc.load_vreg(21, src1)

    # Weight stream does not allocate rd queue entry.
    assert vc.enqueue_scheduler_instruction(
        {"unit": "gsau", "src": 20, "is_weight": True, "expect_output": False}
    )
    # Two result-bearing streams allocate rd entries and should commit in-order.
    assert vc.enqueue_scheduler_instruction(
        {"unit": "gsau", "src": 20, "dst": 30, "is_weight": False}
    )
    assert vc.enqueue_scheduler_instruction(
        {"unit": "gsau", "src": 21, "dst": 31, "is_weight": False}
    )

    _run_core_with_systolic(
        sim,
        eq,
        vc,
        done_cb=lambda: (vc.dump_vreg(30) == [x * 2 for x in src0]) and (vc.dump_vreg(31) == [x * 2 for x in src1]),
        transform_cb=lambda req: [float(x) * 2.0 for x in req["vdata"]],
    )

    assert vc.last_wb is not None
    assert vc.last_wb["source"] == "gsau"
    assert vc.last_wb["dst"] in (30, 31)


if __name__ == "__main__":
    test_vector_core_sim_compute_writeback()
    test_vector_core_sim_vlsu_store_then_load_round_trip()
    test_vector_core_sim_gsau_round_trip_with_rd_queue()
