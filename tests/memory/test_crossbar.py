import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src")))

from base.eventq import EventQueue
from base.clock_domain import ClockDomain
from base.core import Core
from base.sim import Sim

from memory.crossbar import Xbar

def build_sim():
    eq = EventQueue()
    clk = ClockDomain(eq, period=1.0)
    core = Core(eq)
    core.add_clock_domain(clk)
    sim = Sim()
    sim.init(eq, core)
    return eq, clk, sim

def test_crossbar_basic():
    eq, clk, sim = build_sim()

    # Set a small queue size for overflow testing
    x = Xbar(delay=3, num_banks=8, max_size=2)

    results = []
    completions = []

    orig_tick = x.tick

    def tick_and_collect(time=None):
        comp = orig_tick(time)
        if comp:
            completions.extend(comp)
            print(f"[{time}] Completed: {comp}")
        return comp

    x.tick = tick_and_collect
    clk.add_clocked(x)
    clk.schedule_next(0.0)

    def cb(out):
        results.append(out)
        print("[cb] routed output:", out)

    # create a shift mask mapping some inputs to outputs
    shift_mask = [None] * 8
    shift_mask[0] = 3
    shift_mask[1] = 0
    shift_mask[2] = 7
    shift_mask[3] = 1
    # remaining lanes unused (None)

    input_vals = [100 + i for i in range(8)]

    op_id1 = x.enqueue(shift_mask, input_vals, callback=cb)
    op_id2 = x.enqueue(shift_mask, input_vals, callback=cb)
    op_id3 = x.enqueue(shift_mask, input_vals, callback=cb)  # should overflow

    print("submitted xbar ops", op_id1, op_id2, op_id3)
    assert op_id1 > 0 and op_id2 > 0, "First two ops should succeed"
    assert op_id3 == -1, "Third op should fail due to queue overflow"

    sim.run(until=5.0)

    expected = [0] * 8
    expected[3] = input_vals[0]
    expected[0] = input_vals[1]
    expected[7] = input_vals[2]
    expected[1] = input_vals[3]

    assert results, "callback not invoked"
    real_results = [r for r in results if r is not False]
    assert results[0] is False, "First callback should be False due to overflow"
    assert real_results[0] == expected, f"unexpected routed output: {real_results[0]} vs {expected}"

    # verify tick() completion report contains op_id and same output
    assert any(op == op_id1 and out == expected for op, out in completions), f"completion tuple missing or wrong: {completions}"

    stats = x.get_stats()
    print("Xbar stats:", stats)
    assert stats["total_submitted"] >= 2
    assert stats["total_completed"] >= 1

    print("crossbar test passed.")


def test_crossbar_pipeline_staggers_completions():
    x = Xbar(delay=3, num_banks=4)

    shift_mask = [0, 1, 2, 3]
    inputs = [10, 11, 12, 13]

    op1 = x.enqueue(shift_mask, inputs)
    op2 = x.enqueue(shift_mask, inputs)
    op3 = x.enqueue(shift_mask, inputs)

    assert op1 > 0 and op2 > 0 and op3 > 0

    completion_cycles = {}
    for cycle in range(6):
        for op_id, _out in x.tick():
            completion_cycles[op_id] = cycle

    assert completion_cycles[op1] == 3
    assert completion_cycles[op2] == 4
    assert completion_cycles[op3] == 5


def test_crossbar_backpressure_stalls_tail_until_sink_accepts():
    x = Xbar(delay=3, num_banks=4)

    shift_mask = [0, 1, 2, 3]
    inputs = [10, 11, 12, 13]

    attempts = {"op1": 0, "op2": 0}

    def cb1(_out):
        attempts["op1"] += 1
        return attempts["op1"] >= 2

    def cb2(_out):
        attempts["op2"] += 1
        return True

    op1 = x.enqueue(shift_mask, inputs, callback=cb1)
    op2 = x.enqueue(shift_mask, inputs, callback=cb2)

    completion_cycles = {}
    for cycle in range(7):
        for op_id, _out in x.tick():
            completion_cycles[op_id] = cycle

    assert attempts["op1"] == 2
    assert attempts["op2"] == 1
    assert completion_cycles[op1] == 4
    assert completion_cycles[op2] == 5
    assert x.get_stats()["total_retire_stalls"] == 1

if __name__ == "__main__":
    test_crossbar_basic()
    test_crossbar_pipeline_staggers_completions()
    test_crossbar_backpressure_stalls_tail_until_sink_accepts()
