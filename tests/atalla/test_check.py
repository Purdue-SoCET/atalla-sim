"""The dependency checker (src/atalla/check.py)."""
import gzip
import sys
from pathlib import Path

import pytest

from atalla.atalla_platform import build_atalla_platform
from scheduler.assembler import assemble
from scheduler.core import load_program_text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scheduler"))
from test_execute import HALT, base, i, program  # noqa: E402
from test_vector import (DRAM_IN, DRAM_OUT, LOAD_ROWS, meta, sdma, tile_data,  # noqa: E402
                         values, vmem)

PROGRAMS = Path(__file__).resolve().parents[1] / "equivalence" / "programs"


def run(*packets):
    plat = build_atalla_platform(program(*packets, HALT), tile_data(DRAM_IN, LOAD_ROWS, values),
                                 check=True, strict=False)
    plat.run_until_done(limit=200_000)
    return plat


def kinds(plat):
    return {f["kind"] for f in plat.check.report()}


def load(addr_reg, rows=LOAD_ROWS):
    """scpad.ld into pad 0 through x2, then vreg.ld of each row through addr_reg."""
    pk = [*base(1, DRAM_IN), *base(3, meta(0, rows)), i("addi.s", 2, 0, 0),
          i("addi.s", 5, 0, 0), (sdma("scpad.ld", 2, 1, 3),)]
    for rr in reversed(range(rows)):                 # last row first, as gemm loads weights
        pk.append((i("addi.s", 4, 0, rr),))
        pk.append((vmem("vreg.ld", 8 + rr, addr_reg, 4),))
    return pk


def test_a_vreg_ld_through_the_dma_register_is_clean():
    assert kinds(run(*load(addr_reg=2))) == set()


def test_a_vreg_ld_through_another_register_is_a_missing_dependency():
    plat = run(*load(addr_reg=5))
    rows = plat.check.report()
    assert {r["kind"] for r in rows} == {"sdma_then_vreg_ld"}
    assert "the load doesn't read x2" in rows[0]["detail"]
    assert "vreg.ld" in rows[0]["packet"]


def store(sdma_reg):
    """vreg.st two rows into pad 1 through x2, then scpad.st through sdma_reg."""
    pk = load(addr_reg=2) + [i("addi.s", 2, 0, 0), i("addi.s", 6, 0, 0),
                             *base(1, DRAM_OUT), *base(3, meta(1, 2)), i("addi.s", 4, 0, 0),
                             i("addi.s", 9, 0, 1)]
    pk.append((vmem("vreg.st", 8, 2, 4, sid=1),))
    pk.append((vmem("vreg.st", 9, 2, 9, sid=1),))
    pk.append((sdma("scpad.st", sdma_reg, 1, 3),))   # right behind the stores
    return pk


def test_a_scpad_st_on_the_vreg_st_register_waits_and_is_clean():
    assert kinds(run(*store(sdma_reg=2))) == set()


def test_a_scpad_st_on_another_register_is_a_missing_dependency():
    assert kinds(run(*store(sdma_reg=6))) == {"vreg_st_then_sdma_st"}


def test_a_misaligned_word_access_and_a_bad_packet():
    text = "\n".join(["main:",
                      "addi_s x5, x0, 22", "nop", "nop", "nop",
                      "sw_s x6, 0(x5)", "nop", "nop", "nop",
                      "addi_s x7, x0, 1", "addi_s x8, x0, 2", "nop", "nop",
                      "jalr x0, x1, 0", "nop", "nop", "nop"])
    plat = build_atalla_platform(assemble(text).instr, {}, check=True, strict=False)
    try:
        plat.run_until_done(limit=100_000, hang_after=8192)
    except RuntimeError:
        pass                                  # the bad packet may lose a write; not the point
    assert {"misaligned", "packet_rules"} <= kinds(plat)
    assert "uses 0x14" in plat.check.summary()


@pytest.mark.parametrize("name", ["gemm", "gemm_tiled", "softmax", "layernorm_param"])
def test_the_checker_changes_nothing(name):
    instr, data = load_program_text(
        gzip.decompress((PROGRAMS / (name + ".in.gz")).read_bytes()).decode())
    plain = build_atalla_platform(instr, data, strict=False)
    plain.run_until_done()
    checked = build_atalla_platform(instr, data, strict=False, check=True)
    checked.run_until_done()
    assert plain.cycle == checked.cycle
    assert plain.dram._blocks == checked.dram._blocks
