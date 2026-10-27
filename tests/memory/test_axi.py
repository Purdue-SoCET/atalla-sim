"""memory/axi.py: the AXI bus and wrapper, against the cycles derived from the
memory_FA26 RTL (module docstring), with a zero-latency DRAM behind it."""
import gzip
import os
import subprocess
import sys
from pathlib import Path

import pytest

from memory.axi import AXIBus, AXIParams, FixedLatencyDRAM
from memory.backend import Backend
from memory.bus import BusMaster
from memory.dram import DRAM

ROOT = Path(__file__).resolve().parents[2]
PROGRAMS = ROOT / "tests" / "equivalence" / "programs"


def make(latency=0, **params):
    dram = FixedLatencyDRAM(latency)
    return AXIBus(dram, AXIParams(**params)), dram


def run(bus, dram, cycles, start=0, each=None):
    for c in range(start, start + cycles):
        if each:
            each(c)
        bus.tick(float(c))
        dram.tick(float(c))


def test_a_read_burst_comes_back_two_cycles_after_the_dram_and_a_beat_a_cycle():
    """AR in 0, taken in 1, all beats done in 1 (latency 0), beat i from 3 + i."""
    bus, dram = make()
    got = []
    bus.port("sp0").read(0, 0x100, 8, lambda i, c: got.append((i, c)))
    run(bus, dram, 20)
    assert got == [(i, 3 + i) for i in range(8)]
    assert bus.idle


def test_the_dram_latency_adds_on():
    bus, dram = make(latency=10)
    got = []
    bus.port("sp0").read(0, 0x100, 1, lambda i, c: got.append(c))
    run(bus, dram, 20)
    assert got == [1 + 10 + 2]


def test_one_r_beat_a_cycle_for_all_masters_oldest_burst_first():
    bus, dram = make()
    got = []
    for name in ("sp1", "sp0"):
        bus.port(name).read(0, 0x100, 4, lambda i, c, n=name: got.append((n, i, c)))
    run(bus, dram, 20)
    # the arbiter starts at SP0 from idle: SP0's AR goes in cycle 1, SP1's in 2
    assert got == [("sp0", i, 3 + i) for i in range(4)] + [("sp1", i, 7 + i) for i in range(4)]


def test_the_read_arbiter_is_round_robin():
    bus, dram = make()
    order = []
    for name in ("sp0", "sp1", "dcache"):
        for k in range(2):
            bus.port(name).read(0, 0x1000 * k, 1, lambda i, c, n=name: order.append(n))
    run(bus, dram, 30)
    assert order == ["sp0", "sp1", "dcache"] * 2


def test_four_reads_outstanding_per_master():
    bus, dram = make()
    p = bus.port("sp2")
    for k in range(4):
        assert p.can_read()
        p.read(0, 0x40 * k, 1, lambda i, c: None)
    assert not p.can_read()


def test_a_write_burst_goes_out_four_cycles_after_its_aw_and_b_one_after_the_last():
    bus, dram = make()
    taken, done = [], []
    bus.port("sp0").write(0, 0x200, 8, lambda i, c: taken.append((i, c)), done.append)
    run(bus, dram, 30)
    assert taken == [(i, 4 + i) for i in range(8)]       # F = 11
    assert done == [12]
    assert bus.idle


def test_the_next_burst_follows_after_two_or_three_idle_cycles():
    """F + 3 for another master, F + 4 for the same one alone."""
    bus, dram = make()
    first = {}
    bus.port("sp0").write(0, 0x200, 8, lambda i, c: None)
    bus.port("sp1").write(0, 0x400, 8, lambda i, c: first.setdefault("sp1", c))
    run(bus, dram, 40)
    assert first["sp1"] == 11 + 3
    bus, dram = make()
    seen = []
    p = bus.port("sp0")
    p.write(0, 0x200, 8, lambda i, c: None)
    p.write(0, 0x400, 8, lambda i, c: seen.append(c))
    run(bus, dram, 40)
    assert seen[0] == 11 + 4


def test_write_fifos_hold_three_bursts_and_31_beats():
    bus, dram = make()
    p = bus.port("dcache")
    for k in range(3):
        assert p.can_write(8)
        p.write(0, 0x40 * k, 8)
    assert not p.can_write(1)                            # AW FIFO
    bus, dram = make()
    p = bus.port("dcache")
    p.write(0, 0, 16)
    p.write(0, 0x80, 15)
    assert not p.can_write(1)                            # W FIFO


def test_the_icache_cannot_write():
    bus, dram = make()
    with pytest.raises(AssertionError):
        bus.port("icache").write(0, 0, 1)


def test_the_bus_sleeps_when_idle_and_wakes_for_a_late_single_beat():
    bus, dram = make(latency=5)
    got = []
    bus.port("dcache").read(0, 0x40, 1, lambda i, c: got.append(c))
    for c in range(30):
        if bus.next_wake(c - 1) is not None or c == 0:
            bus.tick(float(c))
        dram.tick(float(c))
    assert got == [1 + 5 + 2]


# -- the masters on the bus ---------------------------------------------------------------
def test_a_backend_loads_a_row_per_burst_and_stores_a_row_per_burst():
    bus, ram = make()
    dram, rows = DRAM(), []
    dram.write(0x1000, bytes(range(128)))
    b = Backend(dram_burst_bytes=8, elem_bytes=2, shared_burst_channel=bus.port("sp1"),
                send_sram_write=lambda sp, data, r, tx: rows.append((r, data)) or True,
                send_sram_read=lambda sp, r, tx: bytes([r + 1]) * 64)
    b.attach_dram(dram)
    done = []
    b.driver_to_backend_start_load(0, 0x1000, rows=2, cols=32, callback=done.append)
    run(bus, ram, 40, each=lambda c: b.tick(float(c)))
    assert rows == [(0, bytes(range(64))), (1, bytes(range(64, 128)))]
    assert bus.stats["ar"] == 2 and bus.stats["r_beats"] == 16 and done
    b.driver_to_backend_start_store(0, 0x4000, rows=2, cols=32, callback=done.append)
    run(bus, ram, 60, 40, each=lambda c: b.tick(float(c)))
    assert bus.stats["aw"] == 2 and bus.stats["w_beats"] == 16 and len(done) == 2
    assert dram.read(0x4000, 128) == bytes([1]) * 64 + bytes([2]) * 64


def test_a_cache_line_is_one_burst():
    bus, ram = make()
    p = BusMaster("d", bus.port("dcache"), burst_bytes=64)
    p.read(0x100, 64)
    run(bus, ram, 12, each=lambda c: p.tick(float(c)))
    assert bus.stats["ar"] == 1
    assert all(p.beat_ready(b) for b in range(8))


def frozen(name):
    from scheduler.core import load_program_text
    return load_program_text(gzip.decompress((PROGRAMS / (name + ".in.gz")).read_bytes()).decode())


@pytest.mark.parametrize("name", ["gemm", "relu", "softmax"])
def test_kernels_end_in_the_same_state_through_the_bus(name):
    from atalla.atalla_platform import build_atalla_platform
    instr, data = frozen(name)
    plats = [build_atalla_platform(instr, data, axi=a) for a in (False, True)]
    for p in plats:
        p.run_until_done()
    direct, bus = plats
    assert bus.axi is not None and direct.axi is None
    assert bus.dram._blocks == direct.dram._blocks
    assert [bus.vc.read_vreg(v) for v in range(64)] == [direct.vc.read_vreg(v) for v in range(64)]


def _platform_cycles(mode):
    code = (
        "import sys, gzip; sys.path.insert(0, 'src');"
        "from atalla.atalla_platform import build_atalla_platform;"
        "from scheduler.core import load_program_text;"
        "i, d = load_program_text(gzip.decompress(open("
        "'tests/equivalence/programs/softmax.in.gz', 'rb').read()).decode());"
        "p = build_atalla_platform(i, d, axi=True);"
        "p.run_until_done();"
        "print(p.cycle, p.axi.stats, [q.stats for q in p.axi.ports])"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=dict(os.environ, ATALLA_SCHED=mode), cwd=ROOT)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout


def test_event_and_legacy_scheduling_agree():
    assert _platform_cycles("event") == _platform_cycles("legacy")


def test_the_wrapper_holds_a_limited_number_of_bursts():
    """N_BURST_SLOTS: a burst is taken only while a slot is free; a slot frees
    when its last beat goes out."""
    bus, dram = make(latency=0, burst_slots=2, ar_depth=8)
    p = bus.port("sp0")
    for k in range(4):
        p.read(0, 0x100 * k, 4, lambda i, c: None)
    run(bus, dram, 3)
    assert bus.stats["ar"] == 2 and len(p.reads) == 2
    run(bus, dram, 40, 3)
    assert bus.stats["ar"] == 4 and bus.idle
