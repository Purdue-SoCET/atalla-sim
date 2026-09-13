# tests/test_veggie.py
import pytest

import os, sys

from base.testing import build_sim
from base.eventq import EventQueue
from base.clock_domain import ClockDomain
from base.core import Core
from base.sim import Sim
from vector_core.veggie_file import Veggie, OpBuffer

class IO:
    def __init__(self):
        # Veggie <-> OpBuffer signals
        self.read_reqs = []
        self.write_reqs = []
        self.vreg = {}
        self.vmask = {}
        self.dvalid = {}
        self.mvalid = {}
        self.ready = True
        self.ivalid = []
        # outputs container for op buffer
        self.vreg = {}
        self.vmask = {}


def test_vector_pipeline():
    eq, clk, sim = build_sim()

    veggie = Veggie(bank_count=2, regs_per_bank=8, dread_ports=2, dwrite_ports=1, mask_banks=1)
    opbuf = OpBuffer(num_pairs=1)

    veg_in = IO()
    veg_out = IO()
    op_in = veg_out
    op_out = IO()

    veggie.connect(veg_in, veg_out)
    opbuf.connect(op_in, op_out)

    # register to clock domain (optional here, we schedule ticks explicitly)
    clk.add_clocked(veggie)
    clk.add_clocked(opbuf)

    # schedule a write at t=0.0
    veg_in.write_reqs = [{"port": 0, "bank": 0, "addr": 2, "data": 99}]
    # second operand of the pair, in the other bank so the two can be read
    # in the same cycle
    veggie.data_banks[1][3] = 77
    veg_in.read_reqs = []
    eq.schedule(0.0, veggie.tick, 0.0)

    # schedule a read at t=1.0 (we set the request just before scheduling)
    def place_read(time):
        veg_in.write_reqs = []
        veg_in.read_reqs = [
            {"port": 0, "bank": 0, "addr": 2},
            {"port": 1, "bank": 1, "addr": 3},
        ]
        print(f"[{time}] test: placed read_req")
    eq.schedule(1.0, place_read, 1.0)

    # schedule veggie to service the read shortly after placement
    eq.schedule(1.01, veggie.tick, 1.01)

    # schedule opbuf to sample veggie output after veggie produced it
    eq.schedule(1.02, opbuf.tick, 1.02)

    # inject mask so op buffer can combine it with data
    def inject_mask(time):
        op_in.mvalid = {0: True}
        op_in.vmask = {0: 0xFF}
        print(f"[{time}] Injected mask into op_in")
    eq.schedule(1.03, inject_mask, 1.03)

    # call opbuf again to observe the combined result
    eq.schedule(1.04, opbuf.tick, 1.04)

    # run sim
    sim.run(until=2.0)

    # results
    print("\n--- RESULTS ---")
    print("veg_out.vreg:", veg_out.vreg)
    print("veg_out.dvalid:", veg_out.dvalid)
    print("op_out.ivalid:", op_out.ivalid)
    print("op_out.vreg:", getattr(op_out, "vreg", {}))
    print("op_out.vmask:", getattr(op_out, "vmask", {}))
    print("----------------\n")

    # assertions
    assert veg_out.vreg.get(0, None) == 99, f"Veggie readback failed: {veg_out.vreg}"
    assert op_out.ivalid and op_out.ivalid[0] is True, f"OpBuffer didn't mark ready: {op_out.ivalid}"
    # and it hands over both operands of the pair, then frees the slot
    assert opbuf.slot_ready(0)
    (a, b), mask = opbuf.take(0)
    assert (a, b) == (99, 77) and mask == 0xFF
    assert not opbuf.slot_ready(0), "take() must free the slot"
    print("Test passed")

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


def test_operands_are_collected_through_the_register_file_ports():
    """A datapath instruction reads its sources through the VRF's ports, so a
    pair that lands in one bank serialises instead of arriving together."""
    from vector_core.vector_core import VectorCore

    def cycles_to_writeback(src_a, src_b):
        vc = VectorCore(veggie_size=32 * 16, lane_count=4, dtype="fp16",
                        fu_latencies={"alu": 1})
        vc.write_vreg(src_a, [2.0] * 32, dtype="fp16")
        vc.write_vreg(src_b, [3.0] * 32, dtype="fp16")
        vc.enqueue_scheduler_instruction(
            {"unit": "datapath", "op": "add", "dst": 20,
             "src0": src_a, "src1": src_b})
        for cycle in range(1, 80):
            vc.tick(float(cycle))
            if vc.wb_valid and vc.last_wb and int(vc.last_wb["dst"]) == 20:
                assert vc.read_vreg(20) == [5.0] * 32, "result must still be right"
                return cycle
        raise AssertionError("never wrote back")

    # bank = reg % bank_count (4), so 0 and 4 collide while 0 and 1 do not.
    spread = cycles_to_writeback(0, 1)
    packed = cycles_to_writeback(0, 4)
    assert packed > spread, (
        "a same-bank operand pair must cost more than a spread one: "
        "spread=%d packed=%d" % (spread, packed))


def test_collector_holds_a_partial_pair_until_both_operands_arrive():
    from vector_core.veggie_file import OpBuffer

    ob = OpBuffer(num_pairs=2)
    ob.present(0, [1.0])                 # slot 0, first operand only
    ob.present_mask(0, True)
    assert not ob.slot_ready(0), "one operand is not a pair"
    ob.present(1, [2.0])
    assert ob.slot_ready(0)
    assert not ob.slot_ready(1), "slots are independent"

    (a, b), mask = ob.take(0)
    assert (a, b) == ([1.0], [2.0]) and mask is True
    assert not ob.slot_ready(0), "take() frees the slot"


def test_writeback_buffer_stages_one_entry_per_register_bank():
    """The buffer is the staging point between the units and the VRF: it holds
    results until they can be committed, and never lets two writes to the same
    register bank sit in flight together."""
    from vector_core.vector_core import WBBuffer

    wb = WBBuffer(depth=2)
    wb.start_cycle()
    assert wb.enqueue({"dst": 0, "bank": 0, "data": [1]})
    assert not wb.enqueue({"dst": 4, "bank": 0, "data": [2]}), \
        "a second write to bank 0 in the same cycle must be refused"
    assert wb.enqueue({"dst": 1, "bank": 1, "data": [3]})
    assert not wb.enqueue({"dst": 2, "bank": 2, "data": [4]}), "depth is 2"

    # The reservation is per cycle, but a queued entry keeps holding its bank.
    wb.start_cycle()
    assert not wb.enqueue({"dst": 4, "bank": 0, "data": [5]}), \
        "bank 0 is still occupied by the queued entry"

    assert wb.pop()["bank"] == 0
    wb.start_cycle()
    assert wb.enqueue({"dst": 4, "bank": 0, "data": [6]}), \
        "the bank frees once its entry is committed"


def test_writeback_buffer_is_fifo():
    from vector_core.vector_core import WBBuffer

    wb = WBBuffer(depth=4)
    wb.start_cycle()
    for bank in range(3):
        assert wb.enqueue({"dst": bank, "bank": bank, "data": [bank]})
    assert [wb.pop()["bank"] for _ in range(3)] == [0, 1, 2]
    assert wb.pop() is None
