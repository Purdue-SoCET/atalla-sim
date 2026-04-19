# filepath: /home/asicfab/a/socet149/atalla-sim/tests/test_sc_backend.py
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

def test_sc_backend_load_and_store_flow():
    eq, clk, sim = build_sim()

    writes = []
    stores = []

    def send_sram_write(sp_addr: int, row_bytes: bytes, row_idx: int, tx_id: int) -> bool:
        # accept every write and record it (for loads)
        writes.append((sp_addr, row_bytes, row_idx, tx_id))
        print(f"[send_sram_write] tx={tx_id} row={row_idx} sp_addr={sp_addr} len={len(row_bytes)}")
        return True

    def send_sram_read(sp_addr: int, row_idx: int, tx_id: int) -> bytes:
        # For store: return dummy data for the row
        data = bytes([row_idx + 1] * 6)  # 6 elements, value = row_idx+1
        stores.append((sp_addr, data, row_idx, tx_id))
        print(f"[send_sram_read] tx={tx_id} row={row_idx} sp_addr={sp_addr} len={len(data)}")
        return data

    backend = Backend(
        dram_latency=2, dram_q_depth=8, dram_burst_bytes=4, elem_bytes=1,
        send_sram_write=send_sram_write,
        send_sram_read=send_sram_read
    )

    clk.add_clocked(backend)
    clk.schedule_next(0.0)

    # --- Test LOAD ---
    tx_id = backend.driver_to_backend_start_load(base_sp_addr=100, base_dram_addr=200, rows=2, cols=6)
    print(f"started LOAD tx={tx_id}")

    sim.run(until=8.0)

    print("collected writes (load):", [(w[2], len(w[1]), w[3]) for w in writes])
    assert len(writes) == 2, f"expected 2 sram writes, got {len(writes)}"

    stats = backend.get_stats()
    # 2 rows * 2 subreqs each = 4 DRAM bursts issued
    assert stats["issued_bursts"] == 4, f"expected 4 DRAM bursts issued, got {stats['issued_bursts']}"
    assert stats["tx_completed"] == 1, f"expected tx_completed == 1, got {stats['tx_completed']}"

    # --- Test STORE ---
    writes.clear()
    tx_id2 = backend.driver_to_backend_start_store(base_sp_addr=300, base_dram_addr=400, rows=2, cols=6)
    print(f"started STORE tx={tx_id2}")

    sim.run(until=16.0)

    print("collected stores (store):", [(s[2], len(s[1]), s[3]) for s in stores])
    stats2 = backend.get_stats()
    assert stats2["issued_bursts"] == 8, f"expected 8 DRAM bursts issued after store, got {stats2['issued_bursts']}"
    assert stats2["tx_completed"] == 2, f"expected tx_completed == 2 after store, got {stats2['tx_completed']}"

    print("Backend stats after store:", stats2)
    print("Backend load/store test passed.")

if __name__ == "__main__":
    test_sc_backend_load_and_store_flow()