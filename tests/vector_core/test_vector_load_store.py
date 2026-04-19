import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..//..", "src")))

from base.eventq import EventQueue
from base.clock_domain import ClockDomain
from base.core import Core
from base.sim import Sim
from vector_core.vector_load_store import VLSU


def build_sim():
    eq = EventQueue()
    clk = ClockDomain(eq, period=1.0)
    core = Core(eq)
    core.add_clock_domain(clk)
    sim = Sim()
    sim.init(eq, core)
    return eq, clk, sim


def test_vlsu_load_issue_tracks_destination_per_scratchpad():
    eq, clk, sim = build_sim()
    vls = VLSU(scratchpad_count=2)
    clk.add_clocked(vls)
    clk.schedule_next(0.0)

    assert vls.enqueue_issue({"kind": "load", "scratchpad": 0, "vd": 5, "addr": 0x1000, "dtype": "fp16"})
    assert vls.enqueue_issue({"kind": "load", "scratchpad": 1, "vd": 9, "addr": 0x2000, "dtype": "fp16"})

    sim.run(until=1.0)
    req0 = vls.pop_request()
    assert req0["kind"] == "load"
    assert req0["scratchpad"] == 0
    assert vls.fifo_occupancy() == [1, 0]

    sim.run(until=2.0)
    req1 = vls.pop_request()
    assert req1["kind"] == "load"
    assert req1["scratchpad"] == 1
    assert vls.fifo_occupancy() == [1, 1]
    assert vls.outstanding_loads() == 2


def test_vlsu_load_response_maps_back_to_dest_fifo_order():
    eq, clk, sim = build_sim()
    vls = VLSU(scratchpad_count=2)
    clk.add_clocked(vls)
    clk.schedule_next(0.0)

    assert vls.enqueue_issue({"kind": "load", "scratchpad": 0, "vd": 3, "addr": 0x1000, "dtype": "fp16"})
    assert vls.enqueue_issue({"kind": "load", "scratchpad": 0, "vd": 4, "addr": 0x1040, "dtype": "fp16"})

    sim.run(until=1.0)
    assert vls.pop_request()["kind"] == "load"
    sim.run(until=2.0)
    assert vls.pop_request()["kind"] == "load"
    assert vls.fifo_occupancy() == [2, 0]

    assert vls.push_response({"scratchpad": 0, "data": [11, 12], "addr": 0x1000})
    assert vls.push_response({"scratchpad": 0, "data": [21, 22], "addr": 0x1040})

    sim.run(until=3.0)
    wb0 = vls.pop_writeback()
    assert wb0["vd"] == 3
    assert wb0["data"] == [11, 12]

    sim.run(until=4.0)
    wb1 = vls.pop_writeback()
    assert wb1["vd"] == 4
    assert wb1["data"] == [21, 22]
    assert vls.outstanding_loads(0) == 0


def test_vlsu_store_is_pass_through_and_uses_vrf_read_callback():
    eq, clk, sim = build_sim()
    reads = []

    def _read_vreg(vs):
        reads.append(vs)
        return [100 + vs, 200 + vs]

    vls = VLSU(scratchpad_count=2, read_vreg_cb=_read_vreg)
    clk.add_clocked(vls)
    clk.schedule_next(0.0)
    assert vls.enqueue_issue(
        {
            "kind": "store",
            "scratchpad": 1,
            "vs": 7,
            "addr": 0x3000,
            "swizzle": "col_major",
            "mask": [True, False],
            "dtype": "fp16",
        }
    )

    sim.run(until=1.0)
    req = vls.pop_request()
    assert req["kind"] == "store"
    assert req["scratchpad"] == 1
    assert req["vs"] == 7
    assert req["data"] == [107, 207]
    assert req["swizzle"] == "col_major"
    assert req["mask"] == [True, False]
    assert reads == [7]
    assert vls.outstanding_loads() == 0


def test_vlsu_writeback_callback_consumes_completed_load():
    eq, clk, sim = build_sim()
    writes = []

    def _write_vreg(vd, data):
        writes.append((vd, data))

    vls = VLSU(scratchpad_count=2, write_vreg_cb=_write_vreg)
    clk.add_clocked(vls)
    clk.schedule_next(0.0)

    assert vls.enqueue_issue({"kind": "load", "scratchpad": 1, "vd": 10, "addr": 0x5000, "dtype": "fp16"})
    sim.run(until=1.0)
    _ = vls.pop_request()
    assert vls.outstanding_loads(1) == 1

    assert vls.push_response({"scratchpad": 1, "data": [1, 2, 3, 4]})
    sim.run(until=2.0)

    assert writes == [(10, [1, 2, 3, 4])]
    assert not vls.can_pop_writeback()
    assert vls.outstanding_loads(1) == 0


if __name__ == "__main__":
    test_vlsu_load_issue_tracks_destination_per_scratchpad()
    test_vlsu_load_response_maps_back_to_dest_fifo_order()
    test_vlsu_store_is_pass_through_and_uses_vrf_read_callback()
    test_vlsu_writeback_callback_consumes_completed_load()
