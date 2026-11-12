"""M7: today's kernels, end to end, on the M x N vector core.

The same programs as tests/scheduler/test_kernels.py, run on
build_atalla_platform(vector_arch="fu_groups") -- the M x N core in single
mode with compat reads (plan Q33, Q34) -- and checked the same two ways:
against the functional sim on the identical image, and against each C
program's reference result. Known disagreements are the v1 tables' (strict
xfails): they are compiler or functional-sim issues, not vector-core ones.
"""
import sys
from pathlib import Path

import pytest

from scheduler import golden
from scheduler.core import load_program_text
from atalla.atalla_platform import build_atalla_platform, c_entry_sregs

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS / "scheduler"))
sys.path.insert(0, str(TESTS.parent / "tools"))
import c_kernels      # noqa: E402
import run_kernel     # noqa: E402
import test_kernels as v1k  # noqa: E402  (the v1 kernel tests' known-failure tables)


def run_v2(instr, data, regs):
    plat = build_atalla_platform(instr, data, init_sregs=regs, vector_arch="fu_groups")
    plat.run_until_done(limit=2_000_000)
    return plat


@pytest.mark.golden
@pytest.mark.parametrize("kernel", ["add", "relu", "sigmoid", "layernorm_param", "maxpool",
                                    "softmax", "gemm", "gemm_tiled", "gemms", "conv",
                                    "conv_tiled"])
def test_assembly_kernel_matches_the_functional_sim(kernel, tmp_path):
    instr, data = load_program_text(run_kernel.build_kernel(kernel, [], tmp_path).read_text())
    g = golden.run_golden_state(instr, data, reductions="lanes")
    plat = run_v2(instr, data, None)
    assert golden.differences(plat, g) == []
    assert not plat.core.decode2.violations


_built = {}


def run_c(path: Path):
    if path not in _built:
        try:
            program, data, regs = c_kernels.build(path)
        except c_kernels.CompileError as exc:
            pytest.fail(str(exc))
        _built[path] = (program, data, regs, run_v2(program.instr, data, regs))
    return _built[path]


@pytest.mark.golden
@pytest.mark.parametrize("path", v1k.C_PROGRAMS, ids=lambda p: p.name)
def test_c_program_matches_the_functional_sim(path, request):
    v1k._xfail(request, v1k.DIFFERS_FROM_FUNCTIONAL_SIM, path.stem)
    program, data, regs, plat = run_c(path)
    g = golden.run_golden_state(program.instr, data, reductions="lanes", init_sregs=regs)
    assert golden.differences(plat, g) == []


@pytest.mark.golden
@pytest.mark.parametrize("path", v1k.C_PROGRAMS, ids=lambda p: p.name)
def test_c_program_gives_its_reference_result(path, request):
    reasons = dict(v1k.MISSES_REFERENCE)
    bug = c_kernels.known_bug(path)
    if bug:
        reasons[path.stem] = "known compiler bug: " + bug
    v1k._xfail(request, reasons, path.stem)
    *_, plat = run_c(path)
    ok, message = c_kernels.check(path, plat)
    assert ok, message


def test_v2_platform_uses_the_mxn_core_and_compat_reads():
    plat = build_atalla_platform({0: 0}, vector_arch="fu_groups")
    assert plat.vc.arch == "fu_groups"
    assert plat.vc.cfg.read_mode == "compat"
    assert not plat.core.decode2.vector_operands_at_issue
