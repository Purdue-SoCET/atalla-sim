"""memory/ramulator.py: DRAM timed by Ramulator 2's DDR4 model.

Skipped unless `make ramulator` has built the library. The expected
latencies come from the configurations' timings:
  "rtl"       DDR4-3200W (CL-RCD 20-20, a 4-cycle burst) at two memory
              cycles a core cycle
  "ddr4-1600" DDR4-1600J (CL-RCD-RP 10-10-10, tCCD_L 5) at one
"""
import gzip
import os
import subprocess
import sys
from pathlib import Path

import pytest

from memory.backend import Backend
from memory.bus import BusMaster
from memory.dram import DRAM
from memory.ramulator import DRAM_CONFIGS, RamulatorDRAM, ramulator_available, ramulator_config

pytestmark = pytest.mark.skipif(not ramulator_available(), reason="needs `make ramulator`")

ROOT = Path(__file__).resolve().parents[2]
PROGRAMS = ROOT / "tests" / "equivalence" / "programs"


def run(ram, cycles, start=0, masters=()):
    """Tick the masters, then the DRAM, as the platform's phases do."""
    for c in range(start, start + cycles):
        for m in masters:
            m.tick(float(c))
        ram.tick(float(c))


def lone_read(*args, **kw):
    ram, done = RamulatorDRAM(*args, **kw), []
    assert ram.send(0, 0x1000, 8, False, done.append)
    run(ram, 60)
    assert ram.idle
    return done


def test_a_lone_read_opens_its_row():
    """ACT, RD tRCD later, data CL + burst after that, and one memory cycle
    to arrive: 1 + 10 + 14 = 25 memory cycles at DDR4-1600 (core cycle 24);
    1 + 20 + 24 = 45 at DDR4-3200, two a core cycle (core cycle 22)."""
    assert lone_read("ddr4-1600") == [24]
    assert lone_read("rtl") == [22]


def test_the_rtl_config_is_memory_fa26s():
    ram = RamulatorDRAM("rtl")
    assert ram.ratio == 2 and ram.tx_bytes == 64
    assert ram.params["org"] == "DDR4_8Gb_x16" and ram.params["preset"] == "DDR4_3200W"
    taken = sum(ram.send(0, 0x40 * k, 8, False, lambda c: None) for k in range(300))
    assert taken == 256                      # read_buffer_size


def test_reads_in_one_block_share_one_dram_read():
    """The wrapper's coalescing: reads to a 64-byte block already being read
    complete with it; writes are never merged."""
    ram, done = RamulatorDRAM("rtl"), []
    for c in range(8):
        assert ram.send(c, 0x1000 + 8 * c, 8, False, done.append)
        run(ram, 1, c)
    run(ram, 60, 8)
    assert done == [22] * 8
    assert ram.stats["reads"] == 1 and ram.stats["coalesced"] == 7
    for c in range(4):
        ram.send(100 + c, 0x2000 + 8 * c, 8, True, lambda c: None)
    assert ram.stats["writes"] == 4


def test_reads_in_one_row_follow_every_tccd():
    """Without merging, row hits in one bank group: one RD every tCCD_L = 5
    cycles, however few bytes each moves -- each is a full 64-byte burst."""
    ram, done = RamulatorDRAM("ddr4-1600"), []
    for c in range(8):
        assert ram.send(c, 0x1000 + 8 * c, 8, False, done.append)
        run(ram, 1, c)
    run(ram, 80, 8)
    assert done[0] == 24
    assert [b - a for a, b in zip(done, done[1:])] == [5] * 7
    s = ram.controller_stats()
    assert s["row_hits"] == 7 and s["row_misses"] == 1


def test_a_full_read_queue_refuses():
    ram = RamulatorDRAM("ddr4-1600")
    taken = sum(ram.send(0, 0x40 * k, 8, False, lambda c: None) for k in range(40))
    assert taken == 32 and ram.stats["refused"] == 8


def test_a_burst_larger_than_a_transaction_is_an_error():
    with pytest.raises(ValueError, match="64-byte"):
        RamulatorDRAM().send(0, 0, 128, True, lambda c: None)


@pytest.mark.parametrize("config", ["rtl", "ddr4-1600"])
def test_sleeping_catches_up_refresh(config):
    """Asleep while idle, it runs the skipped memory cycles before the next
    request, so a request after a refresh is timed as if every cycle had
    ticked (tREFI is 7.8 us, 6240 core cycles)."""
    def first_done(tick_every_cycle):
        ram, done = RamulatorDRAM(config), []
        for c in range(13000):
            if c in (6300, 12500):
                ram.send(c, 0x2000 + 8 * c, 8, False, done.append)
            if tick_every_cycle or not ram.idle:
                ram.tick(float(c))
        return done

    assert first_done(True) == first_done(False)


def test_any_value_can_be_overridden():
    """A slower CL makes the lone read slower by as much; a field of the
    config itself changes as well."""
    assert lone_read("ddr4-1600", nCL=12) == [26]
    assert lone_read("rtl", nCL=24) == [24]
    assert lone_read("rtl", coalesce_bytes=0) == [22]
    assert RamulatorDRAM("rtl", read_buffer=64).params["read_buffer"] == 64


def test_the_ddr_cntrl_timings_fold_additive_latency_into_cl():
    """The controller waits tRL = tAL + tCL = 11 after RD: one cycle more than the spec."""
    assert lone_read("ddr_cntrl", nREFI=6240) == [25]


def test_the_ddr_cntrl_refresh_interval_dominates():
    """nREFI 250 with nRFC 172: reads wait behind refreshes most of the time."""
    def mean_latency(config):
        ram, done, sent = RamulatorDRAM(config), [], []
        for c in range(0, 5000, 50):
            sent.append(c)
            ram.send(c, 0x40 * c, 8, False, done.append)
            run(ram, 50, c)
        run(ram, 400, 5000)
        return sum(d - s for d, s in zip(done, sent)) / len(sent)

    assert mean_latency("ddr_cntrl") > 2 * mean_latency("ddr4-1600")


def test_unknown_configs_and_parameters_are_refused():
    with pytest.raises(ValueError, match=", ".join(DRAM_CONFIGS)):
        ramulator_config("fast")
    with pytest.raises(ValueError, match="bogus"):
        ramulator_config(bogus=1)


def test_a_backend_loads_a_tile_through_it():
    rows = []
    ram, dram = RamulatorDRAM(), DRAM()
    dram.write(0x1000, bytes(range(128)))
    b = Backend(dram_burst_bytes=8, elem_bytes=2, shared_burst_channel=ram,
                send_sram_write=lambda sp, data, r, tx: rows.append((r, data)) or True)
    b.attach_dram(dram)
    b.driver_to_backend_start_load(0, 0x1000, rows=2, cols=32)
    run(ram, 400, masters=[b])
    assert rows == [(0, bytes(range(64))), (1, bytes(range(64, 128)))]
    assert ram.stats["reads"] + ram.stats["coalesced"] == 16 and ram.idle


def test_a_cache_port_reads_a_line_through_it():
    ram = RamulatorDRAM("ddr4-1600")
    p = BusMaster("p", ram, burst_bytes=8)
    p.read(0x100, 16)
    run(ram, 24, masters=[p])
    assert not p.beat_ready(0)          # it completes in cycle 24
    run(ram, 1, 24, masters=[p])
    assert p.beat_ready(0) and not p.beat_ready(1)
    run(ram, 10, 25, masters=[p])
    assert p.beat_ready(1) and p.idle


def frozen(name):
    from scheduler.core import load_program_text
    return load_program_text(gzip.decompress((PROGRAMS / (name + ".in.gz")).read_bytes()).decode())


@pytest.mark.parametrize("name", ["gemm", "relu", "softmax"])
def test_kernels_end_in_the_same_state(name):
    """Through the AXI bus to Ramulator, as with the fixed channel: only the
    timing differs."""
    from atalla.atalla_platform import build_atalla_platform
    instr, data = frozen(name)
    plats = [build_atalla_platform(instr, data, dram_channel=ch)
             for ch in (None, RamulatorDRAM())]
    for p in plats:
        p.run_until_done()
    fixed, ram = plats
    assert ram.axi is not None and ram.cycle != fixed.cycle
    assert ram.dram._blocks == fixed.dram._blocks
    assert [ram.vc.read_vreg(v) for v in range(64)] == [fixed.vc.read_vreg(v) for v in range(64)]


def _platform_cycles(mode):
    code = (
        "import sys, gzip; sys.path.insert(0, 'src');"
        "from atalla.atalla_platform import build_atalla_platform;"
        "from memory.ramulator import RamulatorDRAM;"
        "from scheduler.core import load_program_text;"
        "i, d = load_program_text(gzip.decompress(open("
        "'tests/equivalence/programs/gemm.in.gz', 'rb').read()).decode());"
        "p = build_atalla_platform(i, d, dram_channel=RamulatorDRAM());"
        "p.run_until_done();"
        "print(p.cycle, p.dram_channel.stats, p.dram_channel.controller_stats(p.cycle - 1))"
    )
    env = dict(os.environ, ATALLA_SCHED=mode)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout


def test_event_and_legacy_scheduling_agree():
    assert _platform_cycles("event") == _platform_cycles("legacy")
