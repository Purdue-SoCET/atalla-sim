import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src")))

from base.eventq import EventQueue
from base.clock_domain import ClockDomain
from base.core import Core
from base.sim import Sim
from memory.sc_sram_banks import SRAMBanks

def build_sim():
    eq = EventQueue()
    clk = ClockDomain(eq, period=1.0)
    core = Core(eq)
    core.add_clock_domain(clk)
    sim = Sim()
    sim.init(eq, core)
    return eq, clk, sim

def test_sram_staggered_and_stall():
    eq, clk, sim = build_sim()

    # Use queue_size=1 for each bank to test stalls
    banks = SRAMBanks(bank_count=2, bank_size=8, read_latency=2, write_latency=1)
    for b in banks.banks:
        b._pending = b._pending.__class__(max_size=1)
    clk.add_clocked(banks)
    clk.schedule_next(0.0)

    # --- Staggered ops, no stalls ---
    staggered_results = []
    def staggered_cb(data):
        staggered_results.append(data)

    # Write at t=0, read at t=1 (no overlap, so no stall)
    eq.schedule(0.0, lambda t: banks.enqueue_write(2, b"staggered", callback=lambda _: None), 0.0)
    eq.schedule(1.0, lambda t: banks.enqueue_read(2, 9, callback=staggered_cb), 1.0)

    sim.run(until=4.0)

    stats = banks.get_stats()
    print("Staggered stats:", stats)
    assert staggered_results and staggered_results[0] == b"staggered", f"Staggered readback failed: {staggered_results}"
    assert stats["total_enqueue_stalls"] == 0, "Staggered: Expected zero stalls"

    # --- Simultaneous: ops at same time, expect stall ---
    # Reset banks for clean test
    banks = SRAMBanks(bank_count=2, bank_size=8, read_latency=2, write_latency=1)
    for b in banks.banks:
        b._pending = b._pending.__class__(max_size=1)

    stall_results = []
    def stall_cb(data):
        stall_results.append(data)

    # Write and read at t=0 (same bank/address), expect enqueue conflict
    eq, clk, sim = build_sim()
    clk.add_clocked(banks)
    clk.schedule_next(0.0)
    eq.schedule(0.0, lambda t: banks.enqueue_write(2, b"stalltest", callback=lambda _: None), 0.0)
    def expect_conflict(_t):
        try:
            banks.enqueue_read(2, 9, callback=stall_cb)
        except RuntimeError:
            return
        assert False, "Stall: Expected enqueue conflict RuntimeError"
    eq.schedule(0.0, expect_conflict, 0.0)

    sim.run(until=4.0)

    stats = banks.get_stats()
    print("Stall stats:", stats)
    # Only the write should succeed; read enqueue conflicts and should not run
    bank_idx, slot_idx = banks._addr_to_bank_slot(2)
    assert banks.banks[bank_idx].mem[slot_idx] == b"stalltest", "Stall: Write did not commit"
    assert stall_results == [], "Stall: Read should not complete due to stall"
    assert stats["total_enqueue_stalls"] >= 1, "Stall: Expected at least one enqueue stall"

    print("SRAMBanks staggered and stall test passed.")

if __name__ == "__main__":
    test_sram_staggered_and_stall()
