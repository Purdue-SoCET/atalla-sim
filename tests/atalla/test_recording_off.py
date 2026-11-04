"""Recording is opt-in: a plain platform attaches no observer and keeps no
per-packet history, and turning every recorder on changes nothing the
model computes."""
import gzip
from pathlib import Path

import pytest

from atalla.atalla_platform import build_atalla_platform
from base import debug
from scheduler.core import load_program_text

PROGRAMS = Path(__file__).resolve().parents[1] / "equivalence" / "programs"


def frozen(name):
    return load_program_text(gzip.decompress((PROGRAMS / (name + ".in.gz")).read_bytes()).decode())


def final_state(plat):
    return (plat.cycle,
            [plat.core.scalar_reg(r) for r in range(64)],
            [list(plat.vc.read_vreg(v)) for v in range(64)],
            {k: bytes(v) for k, v in plat.dram._blocks.items()})


@pytest.mark.parametrize("name", ["gemm", "layernorm_param"])
def test_recorders_change_nothing(name):
    instr, data = frozen(name)
    plain = build_atalla_platform(instr, data)
    plain.run_until_done()
    full = build_atalla_platform(instr, data, perf=True, perf_timeline=True,
                                 activity=True, history=True)
    full.run_until_done()
    assert final_state(plain) == final_state(full)
    assert plain.core.issued_count == full.core.issued_count == len(full.core.issued)
    assert plain.core.d1d2_count == len(full.core.issued_to_d1d2)


def test_a_plain_platform_records_nothing():
    instr, data = frozen("gemm")
    plat = build_atalla_platform(instr, data)
    plat.run_until_done()
    assert plat.perf is None and plat.activity is None
    assert plat.core.issued == [] and plat.core.issued_to_d1d2 == []
    assert plat.core.scalar_writes == []
    assert plat.core.issued_count > 0
    assert debug.ON is False
