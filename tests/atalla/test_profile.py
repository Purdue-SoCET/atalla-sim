"""The per-packet profile (src/atalla/profile.py)."""
import gzip
from pathlib import Path

from atalla.atalla_platform import build_atalla_platform
from atalla.profile import HALT_ROW
from scheduler.assembler import assemble
from scheduler.core import load_program_text

PROGRAMS = Path(__file__).resolve().parents[1] / "equivalence" / "programs"


def test_the_profile_adds_up_to_the_monitor_and_changes_nothing():
    instr, data = load_program_text(
        gzip.decompress((PROGRAMS / "gemm_tiled.in.gz").read_bytes()).decode())
    plain = build_atalla_platform(instr, data, strict=False)
    plain.run_until_done()
    plat = build_atalla_platform(instr, data, strict=False, profile=True)
    plat.run_until_done()
    assert plat.cycle == plain.cycle
    rows = plat.profile.report()
    assert sum(r["cycles"] for r in rows) == plat.perf.cycles
    assert sum(r["issued"] for r in rows) == plat.perf.report()["cycle_breakdown"]["issue"]
    assert rows == sorted(rows, key=lambda r: (-r["cycles"], r["pc"]))
    assert "where the cycles went, by packet" in plat.profile.summary()


def test_a_divide_charges_its_wait_to_the_dependent_packet_and_names_asm_lines():
    text = "\n".join(["main:",
                      "addi_s x5, x0, 100", "nop", "nop", "nop",
                      "div_s x6, x5, x5", "nop", "nop", "nop",
                      "add_s x7, x6, x6", "nop", "nop", "nop",
                      "jalr x0, x1, 0", "nop", "nop", "nop"])
    prog = assemble(text)
    plat = build_atalla_platform(prog.instr, {}, profile=True, describe_pc=prog.describe,
                                 warm_icache=True)
    plat.run_until_done()
    rows = {r["packet"]: r for r in plat.profile.report()}
    add = rows["add.s x7, x6, x6"]
    assert add["waits"].get("stall:hazard:ex2", 0) > 50 or \
        sum(v for k, v in add["waits"].items() if k.startswith("stall:hazard")) > 50
    assert "asm line 10" in add["where"]
    assert any(r["pc"] == HALT_ROW for r in plat.profile.report())
