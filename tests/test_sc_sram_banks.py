# tests/test_sc_sram_banks.py
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from base.eventq import EventQueue
from base.clock_domain import ClockDomain
from base.core import Core
from base.sim import Sim
from base.clocked_object import Clocked

from scratchpad.sc_sram_banks import SRAMBanks

def build_sim():
    eq = EventQueue()
    clk = ClockDomain(eq, period=1.0)
    core = Core(eq)
    core.add_clock_domain(clk)
    sim = Sim()
    sim.init(eq, core)
    return eq, clk, sim

def test_sc_sram_banks():
    eq, clk, sim = build_sim()

    banks = SRAMBanks(bank_count=4, bank_size=64, read_latency=2, write_latency=1)

    # try registering with clock domain if supported by the model
    try:
        clk.add_clocked(banks)
    except Exception:
        pass

    results = []

    def rcb(data):
        results.append(data)

    # enqueue write then read (ordering preserved in the model)
    eq.schedule(0.0, lambda t: banks.enqueue_write(10, b"hello_world", callback=lambda _: None), 0.0)
    eq.schedule(0.0, lambda t: banks.enqueue_read(10, 11, callback=rcb), 0.0)

    # schedule a few ticks to advance the model and complete ops
    def tick_and_collect(time):
        comp = banks.tick()
        if comp:
            print(f"[{time}] Completed: {comp}")
    eq.schedule(0.1, tick_and_collect, 0.1)
    eq.schedule(1.1, tick_and_collect, 1.1)
    eq.schedule(2.1, tick_and_collect, 2.1)
    eq.schedule(3.1, tick_and_collect, 3.1)

    sim.run(until=4.0)

    # verify data returned
    assert results and results[0] == b"hello_world", f"SRAMBanks readback failed: {results}"

    # collect stall / utilization statistics (work with either API get_stats() or direct attributes)
    try:
        stats = banks.get_stats()
    except Exception:
        per_bank = []
        total_cycles_busy = 0
        total_enqueue_stalls = 0
        for b in getattr(banks, "banks", []):
            cb = getattr(b, "cycles_busy", 0)
            es = getattr(b, "enqueue_stalls", 0)
            total_cycles_busy += cb
            total_enqueue_stalls += es
            per_bank.append({"cycles_busy": cb, "enqueue_stalls": es, "queue_len": len(getattr(b, "_pending", []))})
        stats = {"total_cycles_busy": total_cycles_busy, "total_enqueue_stalls": total_enqueue_stalls, "per_bank": per_bank}

    print("SRAM stats:", stats)

    # basic expectations: banks should have seen at least one busy cycle and at least one enqueue stall
    # (the test enqueues a write and a read to the same address/bank at the same time)
    assert stats["total_cycles_busy"] > 0, "Expected banks to report busy cycles"
    assert stats["total_enqueue_stalls"] >= 1, "Expected at least one enqueue stall"

    print("SRAMBanks test passed.")

if __name__ == "__main__":
    test_sc_sram_banks()