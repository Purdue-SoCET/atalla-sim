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


def _pack_row(values):
    return b"".join(int(value).to_bytes(2, "little") for value in values)


def _read_swizzled_row(spad: Scratchpad, tile_id: int, slot: int) -> bytes:
    tile = spad.tiles[tile_id]
    lane_bytes = []
    for lane in range(spad.num_banks):
        bank = _xor_bank(slot, lane, spad.num_banks)
        lane_bytes.append(tile.banks[bank].mem[slot])
    return b"".join(lane_bytes)


def _write_swizzled_row(spad: Scratchpad, tile_id: int, slot: int, row_bytes: bytes) -> None:
    tile = spad.tiles[tile_id]
    for lane in range(spad.num_banks):
        bank = _xor_bank(slot, lane, spad.num_banks)
        off = lane * spad.elem_bytes
        tile.banks[bank].mem[slot] = row_bytes[off : off + spad.elem_bytes]

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

    dram = DRAM(block_bytes=16)
    backend0 = Backend(dram_latency=2, dram_q_depth=8, dram_burst_bytes=4, elem_bytes=2)
    backend1 = Backend(dram_latency=2, dram_q_depth=8, dram_burst_bytes=4, elem_bytes=2)

    spad.attach_backend(backend0, tile_id=0)
    spad.attach_backend(backend1, tile_id=1)
    backend0.attach_dram(dram)
    backend1.attach_dram(dram)
    clk.add_clocked(backend0)
    clk.add_clocked(backend1)

    backend0_row = _pack_row([200 + i for i in range(32)])
    backend1_row = _pack_row([400 + i for i in range(32)])
    dram.write(0x100, backend0_row)
    dram.write(0x200, backend1_row)

    tx_id0 = backend0.driver_to_backend_start_load(base_sp_addr=20, base_dram_addr=0x100, rows=1, cols=32)
    tx_id1 = backend1.driver_to_backend_start_load(base_sp_addr=4, base_dram_addr=0x200, rows=1, cols=32)
    assert tx_id0 > 0
    assert tx_id1 > 0

    # --- Run simulation ---
    sim.run(until=40)

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

    # --- Check backend writes (load) land in the correct tile-specific slot ---
    assert backend0.get_stats()["tx_completed"] == 1
    assert backend1.get_stats()["tx_completed"] == 1
    assert _read_swizzled_row(spad, tile_id=0, slot=20) == backend0_row
    assert _read_swizzled_row(spad, tile_id=1, slot=4) == backend1_row

    print("Scratchpad frontend/two-backend arbitration and data path test passed.")
    print(spad.get_stats())

    # --- Frontend Read: tile 0 ---
    frontend_reads = []
    def frontend_read_cb(data):
        frontend_reads.append(data)

    # Issue the read (should match what was written above)
    spad.frontends[0].read(10, 0, frontend_read_cb)

    # --- Run simulation ---
    sim.run(until=60)

    # --- Check frontend read result ---
    assert frontend_reads, "No frontend read callback received"
    read_data = b''.join(frontend_reads[0])
    row_bytes = b''.join([(i+1).to_bytes(2, 'little') for i in range(32)])
    assert read_data[:len(row_bytes)] == row_bytes, f"Frontend read data mismatch: {read_data[:len(row_bytes)]} vs {row_bytes}"

    print("Frontend read test passed.")


def test_backends_can_attach_to_scratchpad_slots_and_dram():
    eq, clk, sim = build_sim()
    dram = DRAM(block_bytes=16)
    spad = Scratchpad(num_banks=4, bank_size=16, read_latency=1, write_latency=1, xbar_delay=1, elem_bytes=2)
    backend0 = Backend(dram_latency=1, dram_q_depth=8, dram_burst_bytes=4, elem_bytes=2)
    backend1 = Backend(dram_latency=1, dram_q_depth=8, dram_burst_bytes=4, elem_bytes=2)

    clk.add_clocked(spad)
    clk.add_clocked(backend0)
    clk.add_clocked(backend1)
    clk.schedule_next(0.0)

    spad.attach_backend(backend0, tile_id=0)
    spad.attach_backend(backend1, tile_id=1)
    backend0.attach_dram(dram)
    backend1.attach_dram(dram)

    assert spad.backends == [backend0, backend1]
    assert spad.backend is backend0
    assert backend0.dram is dram
    assert backend1.dram is dram

    load_row0 = _pack_row([1, 2, 3, 4])
    load_row1 = _pack_row([11, 12, 13, 14])
    dram.write(0x100, load_row0)
    dram.write(0x120, load_row1)
    tx_id0 = backend0.driver_to_backend_start_load(base_sp_addr=3, base_dram_addr=0x100, rows=1, cols=4)
    tx_id1 = backend1.driver_to_backend_start_load(base_sp_addr=5, base_dram_addr=0x120, rows=1, cols=4)
    assert tx_id0 > 0
    assert tx_id1 > 0

    sim.run(until=8.0)

    assert _read_swizzled_row(spad, tile_id=0, slot=3) == load_row0
    assert _read_swizzled_row(spad, tile_id=1, slot=5) == load_row1

    store_row0 = _pack_row([21, 22, 23, 24])
    store_row1 = _pack_row([31, 32, 33, 34])
    _write_swizzled_row(spad, tile_id=0, slot=6, row_bytes=store_row0)
    _write_swizzled_row(spad, tile_id=1, slot=7, row_bytes=store_row1)

    tx_id2 = backend0.driver_to_backend_start_store(base_sp_addr=6, base_dram_addr=0x200, rows=1, cols=4)
    tx_id3 = backend1.driver_to_backend_start_store(base_sp_addr=7, base_dram_addr=0x220, rows=1, cols=4)
    assert tx_id2 > 0
    assert tx_id3 > 0

    sim.run(until=16.0)

    assert dram.read(0x200, len(store_row0)) == store_row0
    assert dram.read(0x220, len(store_row1)) == store_row1


def test_frontend_writes_are_not_dropped_when_write_xbar_is_pipelined():
    eq, clk, sim = build_sim()
    spad = Scratchpad(num_banks=8, bank_size=16, read_latency=1, write_latency=2, xbar_delay=3, elem_bytes=2, frontend_queue_size=4)

    clk.add_clocked(spad)
    clk.schedule_next(0.0)

    row0 = _pack_row([10 + i for i in range(8)])
    row1 = _pack_row([30 + i for i in range(8)])

    assert spad.frontend_write(0, row0, row_idx=0, tile_id=0)
    assert spad.frontend_write(1, row1, row_idx=1, tile_id=0)

    sim.run(until=12.0)

    assert _read_swizzled_row(spad, tile_id=0, slot=0) == row0
    assert _read_swizzled_row(spad, tile_id=0, slot=1) == row1

if __name__ == "__main__":
    test_scratchpad_full()
    test_backends_can_attach_to_scratchpad_slots_and_dram()
    test_frontend_writes_are_not_dropped_when_write_xbar_is_pipelined()
