# filepath: /home/asicfab/a/socet149/atalla-sim/tests/test_sc_backend.py
import pytest


from base.testing import build_sim

from memory.backend import Backend


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

def test_a_narrow_tile_store_writes_only_the_tile():
    """The scratchpad hands over a whole 64-byte row; a 4-column tile must put
    only its 8 bytes per row in DRAM, leaving its neighbours alone -- here the
    next tile's columns and the next matrix row, 16 bytes apart."""
    from memory.dram import DRAM
    eq, clk, sim = build_sim()
    rows, cols, stride, base = 4, 4, 16, 0x400

    def send_sram_read(sp_addr, row_idx, tx_id):
        return bytes([0x10 + row_idx] * 64)

    backend = Backend(dram_latency=2, dram_q_depth=8, dram_burst_bytes=32, elem_bytes=2,
                      send_sram_read=send_sram_read)
    dram = backend.attach_dram(DRAM())
    dram.write(base, bytes([0xAA] * 128))
    clk.add_clocked(backend)
    clk.schedule_next(0.0)
    backend.driver_to_backend_start_store(base_sp_addr=0, base_dram_addr=base, rows=rows,
                                          cols=cols, dram_stride=stride)
    sim.run(until=40.0)

    assert backend.get_stats()["tx_completed"] == 1
    for r in range(rows):
        row = dram.read(base + r * stride, stride)
        assert row[:8] == bytes([0x10 + r] * 8), "the tile row"
        assert row[8:] == bytes([0xAA] * 8), "the bytes beside the tile"
    assert dram.read(base + rows * stride, 16) == bytes([0xAA] * 16), "the row below"


def test_a_store_completes_when_the_channel_is_busy():
    """A store row of one burst that can't launch at once is parked and sent
    later. The load path's row handoff must not take the parked write data
    for a loaded row (it did: the DRAM write was lost and the store hung)."""
    from base.clocked_object import Clocked
    from memory.backend import SharedDRAMBurstChannel
    from memory.dram import DRAM
    eq, clk, sim = build_sim()
    channel = SharedDRAMBurstChannel()

    class Hog(Clocked):
        """Another master holding the channel for the first cycles, so the
        store's first row is parked too, then every other cycle."""
        def tick(self, time=None):
            if int(time) < 6 or int(time) % 2 == 0:
                channel.reserve(int(time))

    backend = Backend(dram_latency=2, dram_q_depth=8, dram_burst_bytes=8, elem_bytes=2,
                      shared_burst_channel=channel,
                      send_sram_read=lambda sp, row, tx: bytes([0x20 + row] * 64))
    dram = backend.attach_dram(DRAM())
    clk.add_clocked(Hog())
    clk.add_clocked(backend)
    clk.schedule_next(0.0)
    backend.driver_to_backend_start_store(base_sp_addr=0, base_dram_addr=0x800, rows=4,
                                          cols=4, dram_stride=8)
    sim.run(until=200.0)

    assert backend.get_stats()["tx_completed"] == 1
    assert backend.total_backend_stalls > 0, "the channel was busy at least once"
    for r in range(4):
        assert dram.read(0x800 + 8 * r, 8) == bytes([0x20 + r] * 8)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
