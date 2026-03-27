import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src")))

from base.eventq import EventQueue
from base.clock_domain import ClockDomain
from base.core import Core
from base.sim import Sim

from memory.sc_sram_banks import _xor_bank
from memory.dram import DRAM
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

def test_scratchpad_full():
    eq, clk, sim = build_sim()

    spad = Scratchpad(num_banks=32, bank_size=32, read_latency=2, write_latency=2, xbar_delay=2, elem_bytes=2)
    clk.add_clocked(spad)
    clk.schedule_next(0)

    # --- Frontend Write: tile 0 ---
    row_bytes = b''.join([(i+1).to_bytes(2, 'little') for i in range(32)])
    ok = spad.frontend_write(10, row_bytes, row_idx=0)
    assert ok, "frontend_write failed"

    # --- Frontend Write: tile 1 ---
    row_bytes2 = b''.join([(100+i).to_bytes(2, 'little') for i in range(32)])
    ok2 = spad.frontend_write(40, row_bytes2, row_idx=0)
    assert ok2, "frontend_write (tile 1) failed"

    # --- Backend Write (store): tile 0 ---
    backend_writes = []
    def send_sram_write(sp_addr, row_bytes, row_idx, tx_id):
        backend_writes.append((sp_addr, row_bytes, row_idx, tx_id))
        # Actually write to the scratchpad
        spad._accept_backend_write(sp_addr, row_bytes, row_idx, tx_id)
        return True

    backend = Backend(
        dram_latency=2, dram_q_depth=8, dram_burst_bytes=4, elem_bytes=2,
        send_sram_write=send_sram_write
    )
    clk.add_clocked(backend)

    # Simulate a backend load (DRAM to scratchpad)
    tx_id = backend.driver_to_backend_start_load(base_sp_addr=20, base_dram_addr=1000, rows=1, cols=32)
    # Simulate a backend store (scratchpad to DRAM)
    # For store, backend will request rows from scratchpad; we must respond
    backend_store_rows = []
    def send_sram_read(sp_addr, row_idx, tx_id):
        # Return a row of 0xABCD for test, 32 elements * 2 bytes = 64 bytes
        data = (b'\xAB\xCD' * 32)
        backend_store_rows.append((sp_addr, row_idx, tx_id, data))
        return data
    backend.send_sram_read = send_sram_read
    tx_id2 = backend.driver_to_backend_start_store(base_sp_addr=24, base_dram_addr=2000, rows=1, cols=32)

    # --- Run simulation ---
    sim.run(until=10)

    # --- Check frontend writes ---
    tile0 = spad.tiles[0]
    tile1 = spad.tiles[1]
    vals0 = [tile0.banks[b].mem[10] for b in range(32)]
    expected0 = [b'' for _ in range(32)]
    for lane in range(32):
        bank = _xor_bank(10, lane, 32)
        expected0[bank] = (lane+1).to_bytes(2, 'little')
    assert vals0 == expected0, f"Tile0 slot10 mismatch: {vals0} vs {expected0}"

    vals1 = [tile1.banks[b].mem[8] for b in range(32)]
    expected1 = [b'' for _ in range(32)]
    for lane in range(32):
        bank = _xor_bank(8, lane, 32)
        expected1[bank] = (100+lane).to_bytes(2, 'little')
    assert vals1 == expected1, f"Tile1 slot8 mismatch: {vals1} vs {expected1}"

    # --- Check backend writes (load) ---
    assert len(backend_writes) == 1, f"Expected 1 backend write, got {len(backend_writes)}"
    sp_addr, row_bytes, row_idx, tx_id = backend_writes[0]
    assert sp_addr == 20, f"Backend write sp_addr mismatch: {sp_addr}"
    assert len(row_bytes) == 64, f"Backend write row_bytes length mismatch: {len(row_bytes)}"
    assert row_idx == 0, f"Backend write row_idx mismatch: {row_idx}"

    # --- Check backend store (read) ---
    assert len(backend_store_rows) == 1, f"Expected 1 backend store row, got {len(backend_store_rows)}"
    sp_addr, row_idx, tx_id, data = backend_store_rows[0]
    assert sp_addr == 24, f"Backend store sp_addr mismatch: {sp_addr}"
    assert len(data) == 32*2, f"Backend store row length mismatch: {len(data)}"

    print("Scratchpad frontend/backend arbitration and data path test passed.")
    print(spad.get_stats())

    # --- Frontend Read: tile 0 ---
    frontend_reads = []
    def frontend_read_cb(data):
        frontend_reads.append(data)

    # Issue the read (should match what was written above)
    spad.frontends[0].read(10, 0, frontend_read_cb)

    # --- Run simulation ---
    sim.run(until=20)

    # --- Check frontend read result ---
    assert frontend_reads, "No frontend read callback received"
    read_data = b''.join(frontend_reads[0])
    row_bytes = b''.join([(i+1).to_bytes(2, 'little') for i in range(32)])
    assert read_data[:len(row_bytes)] == row_bytes, f"Frontend read data mismatch: {read_data[:len(row_bytes)]} vs {row_bytes}"

    print("Frontend read test passed.")


def test_backend_can_attach_to_scratchpad_and_dram():
    dram = DRAM(block_bytes=16)
    spad = Scratchpad(num_banks=4, bank_size=16, read_latency=1, write_latency=1, xbar_delay=1, elem_bytes=2)
    backend = Backend(dram_latency=1, dram_q_depth=8, dram_burst_bytes=4, elem_bytes=2)

    spad.attach_backend(backend)
    backend.attach_dram(dram)

    assert spad.backend is backend
    assert backend.dram is dram
    assert backend.send_sram_write == spad._accept_backend_write
    assert backend.send_sram_read == spad.backend_read_row

    load_row = b"".join((i + 1).to_bytes(2, "little") for i in range(4))
    dram.write(0x100, load_row)
    tx_id = backend.driver_to_backend_start_load(base_sp_addr=3, base_dram_addr=0x100, rows=1, cols=4)
    assert tx_id > 0

    for cycle in range(8):
        backend.tick(cycle)
        spad.tick(cycle)

    slot = 3
    got = []
    for lane in range(4):
        bank = _xor_bank(slot, lane, spad.num_banks)
        got.append(spad.tiles[0].banks[bank].mem[slot])
    assert got == [(i + 1).to_bytes(2, "little") for i in range(4)]

    store_row = b"".join((10 + i).to_bytes(2, "little") for i in range(4))
    for lane in range(4):
        bank = _xor_bank(5, lane, spad.num_banks)
        spad.tiles[0].banks[bank].mem[5] = store_row[lane * 2 : lane * 2 + 2]

    tx_id2 = backend.driver_to_backend_start_store(base_sp_addr=5, base_dram_addr=0x200, rows=1, cols=4)
    assert tx_id2 > 0

    for cycle in range(8, 16):
        backend.tick(cycle)
        spad.tick(cycle)

    assert dram.read(0x200, len(store_row)) == store_row

if __name__ == "__main__":
    test_scratchpad_full()
