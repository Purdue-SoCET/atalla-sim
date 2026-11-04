"""Hang diagnosis (src/atalla/hang.py): a run that stops making progress
stops early, with a report of what it waits on."""
import gzip
from pathlib import Path

import pytest

from atalla.atalla_platform import build_atalla_platform
from atalla.hang import SimulationHang, diagnose, progress_signature
from scheduler.assembler import assemble
from scheduler.core import load_program_text

PROGRAMS = Path(__file__).resolve().parents[1] / "equivalence" / "programs"

# Two ops for EX1 in one packet: the hardware runs one, so x6's write is lost
# and the add waits on it forever.
LOST_WRITE = "\n".join(["main:",
                        "addi_s x5, x0, 3", "addi_s x6, x0, 4", "nop", "nop",
                        "add_s x7, x5, x6", "nop", "nop", "nop",
                        "jalr x0, x1, 0", "nop", "nop", "nop"])


def test_a_lost_write_is_reported_as_a_hang_naming_the_register_and_the_packet():
    plat = build_atalla_platform(assemble(LOST_WRITE).instr, {}, strict=False)
    with pytest.raises(SimulationHang) as e:
        plat.run_until_done(limit=1_000_000, hang_after=10_000)
    text = str(e.value)
    assert plat.cycle < 30_000                       # stopped early, not at the limit
    assert "add.s x7, x5, x6" in text
    assert "waiting on x6" in text
    assert "busy registers: scalar x6" in text
    assert "broke the hardware's packet rules" in text
    assert "no block holds work" in text


def test_the_cycle_limit_error_carries_the_same_report():
    plat = build_atalla_platform(assemble(LOST_WRITE).instr, {}, strict=False)
    with pytest.raises(RuntimeError) as e:
        plat.run_until_done(limit=5_000, hang_after=None)
    assert str(e.value).startswith("no halt after 5000 cycles")
    assert "waiting on x6" in str(e.value)


def test_a_working_kernel_never_trips_the_check():
    instr, data = load_program_text(
        gzip.decompress((PROGRAMS / "gemm_tiled.in.gz").read_bytes()).decode())
    plat = build_atalla_platform(instr, data, strict=False)
    plat.run_until_done(hang_after=4096)        # the tightest setting the check allows
    assert plat.core.done


def test_a_mid_run_report_lists_the_blocks_holding_work():
    instr, data = load_program_text(
        gzip.decompress((PROGRAMS / "gemm.in.gz").read_bytes()).decode())
    plat = build_atalla_platform(instr, data, strict=False)
    before = progress_signature(plat)
    for _ in range(600):
        plat.tick()
    assert progress_signature(plat) != before
    text = diagnose(plat)
    assert "blocks holding work:" in text or "decode 2 holds" in text
