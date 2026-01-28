# filepath: /home/asicfab/a/socet149/atalla-sim/tests/test_sc_backend.py
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from base.eventq import EventQueue
from base.clock_domain import ClockDomain
from base.core import Core
from base.sim import Sim

from scratchpad.backend import Backend

def build_sim():
    eq = EventQueue()
    clk = ClockDomain(eq, period=1.0)
    core = Core(eq)
    core.add_clock_domain(clk)
    sim = Sim()
    sim.init(eq, core)
    return eq, clk, sim

def test_sc_backend_load_flow():
    eq, clk, sim = build_sim()

    writes = []

    def send_sram_write(sp_addr: int, row_bytes: bytes, row_idx: int, tx_id: int) -> bool:
        # accept every write and record it
        writes.append((sp_addr, row_bytes, row_idx, tx_id))
        print(f"[send_sram_write] tx={tx_id} row={row_idx} sp_addr={sp_addr} len={len(row_bytes)}")
        return True

    # small latencies / burst sizes so test completes quickly
    backend = Backend(dram_latency=2, dram_q_depth=8, dram_burst_bytes=4, elem_bytes=1, send_sram_write=send_sram_write)

    # register with clock domain
    clk.add_clocked(backend)
    

    # start a load: 2 rows, 6 elements per row -> 2 subreqs per row (ceil(6/4)=2)
    tx_id = backend.start_load(base_sp_addr=100, base_dram_addr=200, rows=2, cols=6)
    print(f"started LOAD tx={tx_id}")

    # drive backend.tick periodically until completion and print status
    def tick_and_reschedule(t, end=5.0, step=0.1):
        backend.tick()
        # print lightweight status
        st = backend.get_stats()
        print(f"[{t:.2f}] dram_pending={st['dram_pending']} issued={st['issued_bursts']} completed={st['completed_bursts']} writes={len(writes)}")
        next_t = t + step
        if next_t <= end:
            eq.schedule(next_t, lambda tt: tick_and_reschedule(tt, end, step), next_t)

    eq.schedule(0.0, lambda t: tick_and_reschedule(t, 3.0, 0.05), 0.0)
    sim.run(until=3.5)

    # Expect one assembled write per row
    print("collected writes:", [(w[2], len(w[1]), w[3]) for w in writes])
    assert len(writes) == 2, f"expected 2 sram writes, got {len(writes)}"

    stats = backend.get_stats()
    # 2 rows * 2 subreqs each = 4 DRAM bursts issued
    assert stats["issued_bursts"] == 4, f"expected 4 DRAM bursts issued, got {stats['issued_bursts']}"
    assert stats["tx_completed"] == 1, f"expected tx_completed == 1, got {stats['tx_completed']}"

    print("Backend stats:", stats)
    print("Backend test passed.")

if __name__ == "__main__":
    test_sc_backend_load_flow()