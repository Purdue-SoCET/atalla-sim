"""Tensors in DRAM (src/atalla/tensors.py), the Kernel API (src/atalla/kernel.py)
and run_kernel's --input / --output / --expect."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from atalla import tensors
from atalla.kernel import Kernel, entry_sregs
from scheduler import golden
from scheduler.assembler import to_in_text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "scheduler"))
from test_execute import HALT, program  # noqa: E402
from test_vector import DRAM_IN, DRAM_OUT, epilogue, prologue, vv  # noqa: E402

ROWS = 4


def doubler():
    """DRAM_IN (4 x 32 BF16) -> pad 0 -> v8..v11 -> doubled -> pad 1 -> DRAM_OUT."""
    pk = prologue(rows=ROWS) + [(vv("add.vv", 20 + k, 8 + k, 8 + k),) for k in range(ROWS)]
    return program(*pk, *epilogue([20 + k for k in range(ROWS)]), HALT)


def x_in():
    return tensors.bf16_round(np.random.default_rng(1).normal(size=(ROWS, 32)))


# -- tensors ------------------------------------------------------------------------------

def test_bf16_rounds_to_nearest_even():
    bits = tensors.to_bf16_bits(np.array([1.0, 1.0 + 2 ** -8, 1.0 + 3 * 2 ** -8, np.nan]))
    assert list(bits[:3]) == [0x3F80, 0x3F80, 0x3F82]          # ties go to even
    assert np.isnan(tensors.from_bf16_bits(bits[3]))


@pytest.mark.parametrize("dtype", ["bf16", "f32", "i32", "u16", "i8"])
def test_put_then_read_round_trips_at_any_alignment(dtype):
    x = np.arange(-5, 6) if dtype != "u16" else np.arange(11)
    data = {0x100: 0xDEADBEEF}
    tensors.put(data, 0x103 if dtype == "i8" else 0x102, x, dtype)
    got = tensors.read(tensors.words_reader(data), 0x103 if dtype == "i8" else 0x102,
                       x.shape, dtype)
    assert np.array_equal(got, x)
    assert data[0x100] & 0xFFFF == 0xBEEF                       # neighbours kept


def test_assert_close_counts_bf16_steps():
    want = np.array([1.0, 2.0, 0.0])
    tensors.assert_close(want + np.array([2 ** -7, 0, 0]), want, ulps=1)
    with pytest.raises(AssertionError, match="1 of 3 elements"):
        tensors.assert_close(want + np.array([2 ** -6, 0, 0]), want, ulps=1)
    m = tensors.compare(want, want)
    assert m["max_abs"] == 0 and m["cosine"] == pytest.approx(1.0)


def test_specs_parse():
    assert tensors.parse_input("a.npy@0x1000") == ("a.npy", 0x1000, "bf16")
    assert tensors.parse_input("dir/a b.npy@4096:f32") == ("dir/a b.npy", 4096, "f32")
    assert tensors.parse_output("0x3000:32x32:bf16=y.npy") == (0x3000, (32, 32), "bf16", "y.npy")
    assert tensors.parse_output("0x3000:8") == (0x3000, (8,), "bf16", None)
    with pytest.raises(ValueError):
        tensors.parse_input("0x1000")


def test_the_stack_goes_above_inputs_and_outputs():
    assert entry_sregs({0x1000: 1}, [(0x8000, 0x8100)])[2] == 0xA000


# -- the Kernel API -----------------------------------------------------------------------

def test_a_kernel_from_a_program_with_numpy_inputs():
    x = x_in()
    r = Kernel(program=(doubler(), {})).write(DRAM_IN, x).run(check=True)
    tensors.assert_close(r.read(DRAM_OUT, (ROWS, 32)), 2 * x, ulps=0)
    assert r.cycles > 0 and not r.plat.check.findings


@pytest.mark.golden
@pytest.mark.skipif(not golden.HAVE_GOLDEN, reason="no functional sim")
def test_golden_runs_the_same_image_on_the_functional_sim():
    x = x_in()
    r = Kernel(program=(doubler(), {})).write(DRAM_IN, x).run(golden=True)
    assert np.array_equal(r.golden.read(DRAM_OUT, (ROWS, 32)), r.read(DRAM_OUT, (ROWS, 32)))


def test_exactly_one_source():
    with pytest.raises(ValueError):
        Kernel()


# -- the command line ---------------------------------------------------------------------

def test_run_kernel_input_output_expect(tmp_path):
    x = x_in()
    np.save(tmp_path / "x.npy", x)
    np.save(tmp_path / "want.npy", 2 * x)
    np.save(tmp_path / "wrong.npy", 2 * x + 1)
    (tmp_path / "doubler.in").write_text(to_in_text(doubler(), {}))

    def run(*extra):
        return subprocess.run([sys.executable, str(ROOT / "tools" / "run_kernel.py"),
                               str(tmp_path / "doubler.in"),
                               "--input", "%s@%#x" % (tmp_path / "x.npy", DRAM_IN), *extra],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True)

    p = run("--output", "%#x:%dx32=%s" % (DRAM_OUT, ROWS, tmp_path / "y.npy"),
            "--expect", "%s@%#x" % (tmp_path / "want.npy", DRAM_OUT))
    assert p.returncode == 0, p.stdout
    assert "PASS" in p.stdout
    assert np.array_equal(np.load(tmp_path / "y.npy"), 2 * x)
    p = run("--expect", "%s@%#x" % (tmp_path / "wrong.npy", DRAM_OUT))
    assert p.returncode == 1 and "FAIL" in p.stdout
