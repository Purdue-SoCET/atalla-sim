import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src")))

from base.eventq import EventQueue
from base.clock_domain import ClockDomain
from base.core import Core
from base.sim import Sim

from memory.scratchpad import Scratchpad
from memory.backend import Backend

def build_sim():
    eq = EventQueue()
    clk = ClockDomain(eq, period=1.0)
    core = Core(eq)
    core.add_clock_domain(clk)
    sim = Sim()
    sim.init(eq, core)
    return eq, clk, sim

def test_scratchpad_stalls():
    eq, clk, sim = build_sim()

    # Use small frontend queue size to force stalls
    spad = Scratchpad(num_banks=8, bank_size=8, read_latency=2, write_latency=2, xbar_delay=2, elem_bytes=2, frontend_queue_size=2)
    clk.add_clocked(spad)
    clk.schedule_next(0)

    # --- Frontend vs Frontend: Fill write queue, then overflow ---
    row_bytes = b''.join([(i+1).to_bytes(2, 'little') for i in range(8)])
    ok1 = spad.frontend_write(2, row_bytes, row_idx=0, tile_id=0)
    ok2 = spad.frontend_write(3, row_bytes, row_idx=1, tile_id=0)
    ok3 = spad.frontend_write(4, row_bytes, row_idx=2, tile_id=0)  # Should stall
    assert ok1 and ok2, "First two frontend writes should succeed"
    assert not ok3, "Third frontend write should stall (queue full)"

    stats = spad.get_stats()
    print("Frontend vs Frontend stats:", stats)
    assert stats["frontend_stalls"][0]["write_stalled"], "Frontend write stall flag not set"

    # --- Backend vs Backend: Fill backend queue, then overflow ---
    backend_writes = []
    def send_sram_write(sp_addr, row_bytes, row_idx, tx_id):
        backend_writes.append((sp_addr, row_bytes, row_idx, tx_id))
        spad._accept_backend_write(sp_addr, row_bytes, row_idx, tx_id)
        return True

    backend = Backend(
        dram_latency=2, dram_q_depth=2, dram_burst_bytes=4, elem_bytes=2,
        send_sram_write=send_sram_write
    )
    clk.add_clocked(backend)

    # This will attempt 4 bursts for one row, but only 2 can be pending
    tx_id = backend.driver_to_backend_start_load(base_sp_addr=5, base_dram_addr=1000, rows=1, cols=8)
    sim.run(until=10.0)
    backend_stats = backend.get_stats()
    print("Backend vs Backend stats:", backend_stats)
    assert backend_stats["backend_stalls"] >= 2, "Backend stalls not detected"

    # --- Frontend vs Backend: Backend in flight blocks frontend ---
    # Fill backend write inflight flag manually to simulate contention
    spad.backend_write_inflight[0] = True
    ok4 = spad.frontend_write(6, row_bytes, row_idx=3, tile_id=0)
    assert ok4, "Frontend write should enqueue (queue not full)"
    # But tick will not process it until backend_write_inflight is cleared
    spad.frontends[0].tick(2)
    # The request should remain in the queue
    assert len(spad.frontends[0].writeq.items) == 1, "Frontend write should be blocked by backend inflight"

    # Now clear backend inflight and tick again
    spad.backend_write_inflight[0] = False
    spad.frontends[0].tick(10)
    # The request should be processed
    assert len(spad.frontends[0].writeq.items) == 0, "Frontend write should be processed after backend inflight cleared"

    print("Scratchpad stall test passed.")

if __name__ == "__main__":
    test_scratchpad_stalls()