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

    # try registering with clock domain if supported by the model
    try:
        clk.add_clocked(x)
    except Exception:
        pass

    results = []
    completions = []

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

    def tick_and_collect(time):
        comp = x.tick()
        if comp:
            completions.extend(comp)
            print(f"[{time}] Completed: {comp}")

    eq.schedule(0.1, tick_and_collect, 0.1)
    eq.schedule(1.1, tick_and_collect, 1.1)
    eq.schedule(2.1, tick_and_collect, 2.1)
    eq.schedule(3.1, tick_and_collect, 3.1)
    eq.schedule(4.1, tick_and_collect, 4.1)

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

if __name__ == "__main__":
    test_crossbar_basic()
