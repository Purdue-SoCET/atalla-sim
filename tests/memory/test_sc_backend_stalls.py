import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src")))

from base.eventq import EventQueue
from base.clock_domain import ClockDomain
from base.core import Core
from base.sim import Sim

from memory.backend import Backend

def build_sim():
    eq = EventQueue()
    clk = ClockDomain(eq, period=1.0)
    core = Core(eq)
    core.add_clock_domain(clk)
    sim = Sim()
    sim.init(eq, core)
    return eq, clk, sim

def test_backend_stalls():
    eq, clk, sim = build_sim()

    writes = []
    stores = []

    def send_sram_write(sp_addr: int, row_bytes: bytes, row_idx: int, tx_id: int) -> bool:
        writes.append((sp_addr, row_bytes, row_idx, tx_id))
        return True

    def send_sram_read(sp_addr: int, row_idx: int, tx_id: int) -> bytes:
        data = bytes([row_idx + 1] * 32)  # 32 elements, value = row_idx+1
        stores.append((sp_addr, data, row_idx, tx_id))
        return data

    # Set dram_q_depth=2 so we can force stalls
    backend = Backend(
        dram_latency=2, dram_q_depth=2, dram_burst_bytes=4, elem_bytes=1,
        send_sram_write=send_sram_write,
        send_sram_read=send_sram_read
    )

    clk.add_clocked(backend)
    clk.schedule_next(0.0)

    # --- Test LOAD with guaranteed stalls ---
    # 1 row, 32 cols, elem_bytes=1, burst_bytes=4 => 1 row * 8 subreqs/row = 8 bursts
    # Only 2 can be pending, so 6 will stall on the first tick
    tx_id = backend.driver_to_backend_start_load(base_sp_addr=100, base_dram_addr=200, rows=1, cols=32)
    print(f"started LOAD tx={tx_id}")

    sim.run(until=16.0)

    stats = backend.backend_to_driver_get_stats()
    print("Backend stats after forced stalls:", stats)
    # There should be at least 6 stalls (8 bursts attempted, 2 accepted, 6 stalled)
    assert stats["backend_stalls"] >= 6, f"Expected at least 6 backend stalls, got {stats['backend_stalls']}"
    print("Backend stall test passed.")

if __name__ == "__main__":
    test_backend_stalls()