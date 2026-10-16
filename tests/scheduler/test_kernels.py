"""Stage 6: real kernels, end to end on the scheduler-driven platform.

Two kinds: assembly kernels (here) and C programs (further down, straight
from the compiler). Each assembly kernel is assembled by the functional sim's own build script
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
import sys
from pathlib import Path

import pytest

from scheduler import golden
from scheduler.core import load_program_text
from atalla.atalla_platform import build_atalla_platform

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import run_kernel  # noqa: E402  (tools/run_kernel.py: builds the kernels)


def build(kernel: str, tmp_path: Path) -> str:
    """An assembly kernel's program text (functional sim build script)."""
    return run_kernel.build_kernel(kernel, [], tmp_path).read_text()


def run_both(text: str, reductions: str = "lanes", c_kernel: bool = False):
    """Run a program on the platform and on the functional sim; returns
    (platform, functional sim state). C kernels start with the C loader's
    stack registers, in both."""
    from atalla.atalla_platform import c_entry_sregs
    instr, data = load_program_text(text)
    regs = c_entry_sregs(data) if c_kernel else None
    g = golden.run_golden_state(instr, data, reductions=reductions, init_sregs=regs)
    plat = build_atalla_platform(instr, data, init_sregs=regs)
    plat.run_until_done(limit=2_000_000)
    return plat, g


@pytest.mark.golden
@pytest.mark.parametrize("kernel", ["add", "relu", "sigmoid", "layernorm_param", "maxpool",
                                    "softmax", "gemm", "gemm_tiled", "gemms", "conv",
                                    "conv_tiled"])
def test_kernel_matches_the_functional_sim(kernel, tmp_path):
    plat, g = run_both(build(kernel, tmp_path))
    assert golden.differences(plat, g) == []
    assert not plat.core.decode2.violations


@pytest.mark.golden
def test_softmax_against_the_functional_sims_own_sum(tmp_path):
    """softmax sums with rsum.vi. The functional sim adds in fp32, in
    element order, and keeps the sum unrounded; the lanes add per lane and
    then across lanes, and the register file rounds to BF16. The sum's
    reciprocal scales every output, so outputs may differ by a BF16 step or
    two, and nothing more."""
    plat, g = run_both(build("softmax", tmp_path), reductions="functional")
    off = [abs(m - r) for m, r in golden.dram_halfwords(plat, g).values() if m != r]
    assert off and max(off) <= 2


# -- C programs ---------------------------------------------------------------------
# Straight from the compiler: atalla_cc -p packetizes, src/scheduler/assembler.py
# assembles, and the inputs and reference results come from the compiler's own
# validation harness (tools/c_kernels.py). Each program is checked two ways:
# against the functional sim running the identical image, and against its
# reference result (numpy, or x10 for the unit tests).
#
# Known disagreements are strict xfails: when one is fixed the test reports
# XPASS and fails, so remove its entry here.

import c_kernels  # noqa: E402  (tools/c_kernels.py)

C_PROGRAMS = (sorted(c_kernels.KERNELS.glob("*.c")) + sorted(c_kernels.UNIT_TESTS.glob("*.c"))
              if (c_kernels.COMPILER / "ppci").is_dir() else [])

#: vector_load() compiles to vreg.ld straight from a DRAM address. The functional
#: sim special-cases that (src/components/scpad_ls.py, addresses >= 0x1000 read
#: DRAM); the hardware's vreg.ld only reads the scratchpad. A compiler issue.
_DRAM_SHORTCUT = "vreg.ld/st on a DRAM address: only the functional sim allows it"
#: vreg.ld/st move all 32 lanes where the functional sim moves num_cols + 1, so
#: a register loaded from a 4-wide tile holds different lanes 4..31.
_NUM_COLS = "vreg.ld/st move 32 lanes; the functional sim moves num_cols + 1"
#: sw.s to an address that is not a multiple of 4: the data cache aligns it
#: down (word-addressed memory), the functional sim writes at the byte address.
#: A compiler bug: the vector load doesn't wait for the DMA that fills its
#: scratchpad. Only a read of the DMA's rs1 waits for it, and the compiler
#: puts the address in a different register. Correct only when the DMA wins.
_DMA_RACE = "vector_load doesn't wait for its scpad_load (no shared register)"
_MISALIGNED = "misaligned sw.s on the stack: aligned down here, not in the functional sim"

DIFFERS_FROM_FUNCTIONAL_SIM = {
    **{n: _DRAM_SHORTCUT for n in (
        "gemm_passthrough_lane0", "load_weights_lane0_dot", "make_mask_eq_vector",
        "make_mask_gt_scalar", "scpad_store_roundtrip", "vec_op_masked_add_scalar_lane0",
        "vec_op_masked_mul_vector_lane0", "vector_add_lane0", "vector_store_roundtrip")},
    "gemm_tiled_baseline": _NUM_COLS,
    "layernorm": _MISALIGNED,
    "scpad_load_roundtrip": _DMA_RACE,
}

#: Why a program misses its reference result (besides the compiler's own
#: EXPECTED_FAILURES list, which c_kernels.known_bug reads).
_WEIGHT_ORDER = ("the kernel loads weight rows first to last; lw.vi shifts each into "
                 "column 0, so tiles come out column-reversed")
MISSES_REFERENCE = {
    **{n: _WEIGHT_ORDER for n in ("conv_baseline", "conv_pipelined", "conv_pipelined_unrolled",
                                  "gemm_tiled_baseline", "gemm_tiled_pipelined",
                                  "gemm_tiled_pipelined_unrolled")},
    **{n: _DRAM_SHORTCUT for n in DIFFERS_FROM_FUNCTIONAL_SIM
       if DIFFERS_FROM_FUNCTIONAL_SIM[n] == _DRAM_SHORTCUT and n != "make_mask_eq_vector"},
    "scpad_load_roundtrip": _DMA_RACE,
}

_built = {}


def run_c(path: Path):
    """Compile, assemble and run once per program; (program, data, regs, platform)."""
    if path not in _built:
        try:
            program, data, regs = c_kernels.build(path)
        except c_kernels.CompileError as exc:
            pytest.fail(str(exc))
        plat = build_atalla_platform(program.instr, data, init_sregs=regs)
        plat.run_until_done(limit=2_000_000)
        _built[path] = (program, data, regs, plat)
    return _built[path]


def _xfail(request, reasons, name):
    if name in reasons:
        request.applymarker(pytest.mark.xfail(reason=reasons[name], strict=True))


@pytest.mark.golden
@pytest.mark.parametrize("path", C_PROGRAMS, ids=lambda p: p.name)
def test_c_program_matches_the_functional_sim(path, request):
    _xfail(request, DIFFERS_FROM_FUNCTIONAL_SIM, path.stem)
    program, data, regs, plat = run_c(path)
    g = golden.run_golden_state(program.instr, data, reductions="lanes", init_sregs=regs)
    assert golden.differences(plat, g) == []


@pytest.mark.golden
@pytest.mark.parametrize("path", C_PROGRAMS, ids=lambda p: p.name)
def test_c_program_gives_its_reference_result(path, request):
    reasons = dict(MISSES_REFERENCE)
    bug = c_kernels.known_bug(path)
    if bug:
        reasons[path.stem] = "known compiler bug: " + bug
    _xfail(request, reasons, path.stem)
    *_, plat = run_c(path)
    assert not plat.core.decode2.violations, "the compiler made a packet the RTL can't run"
    ok, message = c_kernels.check(path, plat)
    assert ok, message
