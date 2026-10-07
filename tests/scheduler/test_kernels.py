"""Stage 6: real kernels, end to end on the scheduler-driven platform.

Each kernel is assembled by the functional sim's own build script
(third_party/atalla-functional-sim/kernels/build_*.py; the assembler expands
li.s, and kernels load weight rows last to first because lw.vi shifts a
column in at column 0), run on the platform
(scheduler, vector core, scratchpads and DMA, DRAM) and on the functional
sim, and compared at the end: every scalar and vector register, and DRAM as
the DMA reads it (each BF16 halfword through the functional sim's
read_bf16_le; its data memory is keyed by address, with overlapping 32-bit
entries).

The units compute every value: the lane datapath the element ops and
reductions, the systolic array gemm.vv. The functional sim runs with its
reductions in the lane model's order (golden.lanes_reduce), which is what
the lane datapath does; its own sequential fp32 sum differs slightly, which
the last test bounds.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scheduler import golden
from scheduler.core import load_program_text
from atalla.atalla_platform import build_atalla_platform
from scheduler.semantics import u32
from scheduler.vector import bf16_bits, bf16_round

SIM = golden.SUBMODULE_PATH


def build(kernel: str, tmp_path: Path) -> str:
    golden.require(pytest)
    out = tmp_path / ("%s.in" % kernel)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(SIM), str(SIM / "kernels"), str(SIM / "experiments" / "gemm")]))
    p = subprocess.run([sys.executable, str(SIM / "kernels" / ("build_%s.py" % kernel)),
                        "-o", str(out)], cwd=str(SIM), env=env,
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    return out.read_text()


def run_both(text: str, reductions: str = "lanes"):
    instr, data = load_program_text(text)
    g = golden.run_golden_state(instr, data, reductions=reductions)
    plat = build_atalla_platform(instr, data)
    plat.run_until_done(limit=2_000_000)
    from src.misc.memory import Memory              # the functional sim's
    gm = Memory()
    gm.data_mem = dict(g["mem"])
    halves = sorted({(a & ~3) + k for a in g["mem"] for k in (0, 2)})
    dram = {b: (int.from_bytes(plat.dram.read(b, 2), "little"), gm.read_bf16_le(b))
            for b in halves}
    return plat, g, dram


def assert_registers_match(plat, g, skip=()):
    c = plat.core
    for r in range(1, 256):
        if r in skip:
            continue
        assert c.scalar_reg(r) == u32(g["sregs"].get(r, 0)), "x%d" % r
    for v in range(256):
        assert plat.vc.read_vreg(v) == [bf16_round(x) for x in g["vregs"][v]], "v%d" % v


@pytest.mark.parametrize("kernel", ["add", "relu", "sigmoid", "layernorm_param", "maxpool",
                                    "softmax", "gemm", "gemm_tiled", "gemms", "conv",
                                    "conv_tiled"])
def test_kernel_matches_the_functional_sim(kernel, tmp_path):
    plat, g, dram = run_both(build(kernel, tmp_path))
    assert_registers_match(plat, g)
    bad = {hex(b): v for b, v in dram.items() if v[0] != v[1]}
    assert not bad
    assert not plat.core.decode2.violations


def test_softmax_against_the_functional_sims_own_sum(tmp_path):
    """softmax sums with rsum.vi. The functional sim adds in fp32, in
    element order, and keeps the sum unrounded; the RTL adds pairwise in
    BF16. The sum's reciprocal scales every output, so outputs may differ
    by a BF16 step or two, and nothing more."""
    plat, g, dram = run_both(build("softmax", tmp_path), reductions="functional")
    off = [abs(m - r) for m, r in dram.values() if m != r]
    assert off and max(off) <= 2


# -- C kernels ----------------------------------------------------------------------
# Compiled with atalla_cc (third_party/aihw-ppci-compiler), packetized and given
# their DRAM data by the functional sim's own C runner, and run with the C
# loader's stack registers -- tools/run_kernel.py does the same.

COMPILER = Path(__file__).resolve().parents[2] / "third_party" / "aihw-ppci-compiler"


def build_c(name: str, tmp_path: Path) -> str:
    golden.require(pytest)
    kernels = COMPILER / "atalla_tests" / "kernels"
    if not (kernels / name).is_file():
        pytest.skip("compiler submodule (with atalla_tests/kernels) not checked out")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    import run_kernel
    return run_kernel.build_c_kernel(kernels / name, [], tmp_path).read_text()


@pytest.mark.parametrize("kernel", ["add_4x32.c", "softmax_row32.c", "layernorm_4x4_active.c"])
def test_c_kernel_matches_the_functional_sim(kernel, tmp_path):
    from atalla.atalla_platform import c_entry_sregs
    instr, data = load_program_text(build_c(kernel, tmp_path))
    regs = c_entry_sregs(data)
    g = golden.run_golden_state(instr, data, reductions="lanes", init_sregs=regs)
    plat = build_atalla_platform(instr, data, init_sregs=regs)
    plat.run_until_done(limit=2_000_000)
    assert_registers_match(plat, g)
    from src.misc.memory import Memory
    gm = Memory()
    gm.data_mem = dict(g["mem"])
    bad = [b for b in sorted({(a & ~3) + k for a in g["mem"] for k in (0, 2)})
           if int.from_bytes(plat.dram.read(b, 2), "little") != gm.read_bf16_le(b)]
    assert not bad, [hex(b) for b in bad[:8]]
    assert not plat.core.decode2.violations, "the packetizer made a packet the RTL can't run"
