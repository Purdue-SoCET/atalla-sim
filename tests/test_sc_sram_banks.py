import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from base.eventq import EventQueue
from base.clock_domain import ClockDomain
from base.core import Core
from base.sim import Sim
from scratchpad.sc_sram_banks import SRAMBanks

def build_sim():
    eq = EventQueue()
    clk = ClockDomain(eq, period=1.0)
    core = Core(eq)
    core.add_clock_domain(clk)
    sim = Sim()
    sim.init(eq, core)
    return eq, clk, sim

def test_sram_lockstep_and_stall():
    eq, clk, sim = build_sim()

    # Use queue_size=1 for each bank to test stalls
    banks = SRAMBanks(bank_count=2, bank_size=8, read_latency=2, write_latency=1)
    for b in banks.banks:
        b._pending = b._pending.__class__(max_size=1)

    # --- Lockstep: staggered ops, no stalls ---
    lockstep_results = []
    def lockstep_cb(data):
        lockstep_results.append(data)

    # Write at t=0, read at t=1 (no overlap, so no stall)
    eq.schedule(0.0, lambda t: banks.enqueue_write(2, b"lockstep", callback=lambda _: None), 0.0)
    eq.schedule(1.0, lambda t: banks.enqueue_read(2, 8, callback=lockstep_cb), 1.0)

    def tick_and_collect(time):
        banks.tick()
    eq.schedule(0.1, tick_and_collect, 0.1)
    eq.schedule(1.1, tick_and_collect, 1.1)
    eq.schedule(2.1, tick_and_collect, 2.1)
    eq.schedule(3.1, tick_and_collect, 3.1)

    sim.run(until=4.0)

    stats = banks.get_stats()
    print("Lockstep stats:", stats)
    assert lockstep_results and lockstep_results[0] == b"lockstep", f"Lockstep readback failed: {lockstep_results}"
    assert stats["total_enqueue_stalls"] == 0, "Lockstep: Expected zero stalls"

    # --- Simultaneous: ops at same time, expect stall ---
    # Reset banks for clean test
    banks = SRAMBanks(bank_count=2, bank_size=8, read_latency=2, write_latency=1)
    for b in banks.banks:
        b._pending = b._pending.__class__(max_size=1)

    stall_results = []
    def stall_cb(data):
        stall_results.append(data)

    # Write and read at t=0 (same bank/address), expect read to stall
    eq, clk, sim = build_sim()
    eq.schedule(0.0, lambda t: banks.enqueue_write(2, b"stalltest", callback=lambda _: None), 0.0)
    eq.schedule(0.0, lambda t: banks.enqueue_read(2, 9, callback=stall_cb), 0.0)

    eq.schedule(0.1, tick_and_collect, 0.1)
    eq.schedule(1.1, tick_and_collect, 1.1)
    eq.schedule(2.1, tick_and_collect, 2.1)
    eq.schedule(3.1, tick_and_collect, 3.1)

    sim.run(until=4.0)

    stats = banks.get_stats()
    print("Stall stats:", stats)
    # Only the write should succeed, read should not be enqueued
    assert stall_results == [], "Stall: Read should not complete due to stall"
    assert stats["total_enqueue_stalls"] >= 1, "Stall: Expected at least one enqueue stall"

    print("SRAMBanks lockstep and stall test passed.")

if __name__ == "__main__":
    test_sram_lockstep_and_stall()