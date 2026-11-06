"""The `atalla-sim` command (src/atalla/cli.py) and `atalla-sim doctor`."""
import gzip
import subprocess
import sys
from pathlib import Path

import pytest

from atalla import cli, doctor

ROOT = Path(__file__).resolve().parents[2]


def atalla_sim(*args, env=None):
    return subprocess.run([sys.executable, "-m", "atalla.cli", *args], cwd=str(ROOT),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True,
                          env=dict(__import__("os").environ, PYTHONPATH=str(ROOT / "src"),
                                   **(env or {})))


def test_usage_lists_every_command():
    p = atalla_sim("--help")
    assert p.returncode == 0
    for name in cli.COMMANDS:
        assert name in p.stdout


def test_an_unknown_command_says_so():
    p = atalla_sim("frobnicate")
    assert p.returncode != 0 and "unknown command 'frobnicate'" in p.stdout


def test_run_prints_only_the_cycle_count(tmp_path):
    prog = tmp_path / "relu.in"
    prog.write_bytes(gzip.decompress((ROOT / "tests/equivalence/programs/relu.in.gz").read_bytes()))
    p = atalla_sim("run", str(prog))
    assert p.returncode == 0, p.stdout
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1 and "relu.in: " in lines[0] and " cycles " in lines[0]


def test_doctor_passes_here_and_names_a_fix_for_what_is_missing():
    checks = doctor.run_checks()
    by = {c.name: c for c in checks}
    assert by["python"].status == doctor.OK and by["numpy"].status == doctor.OK
    for c in checks:
        if c.status != doctor.OK:
            assert c.fix, c
    text = doctor.render(checks)
    assert "ready to simulate" in text


def test_doctor_reports_a_missing_native_library_as_optional():
    p = atalla_sim("doctor", env={"ATALLA_NO_NATIVE": "1"})
    assert p.returncode == 0, p.stdout
    assert "make native" in p.stdout


def test_an_old_python_gets_a_message_not_a_traceback(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 6, 8))
    with pytest.raises(SystemExit) as e:
        cli.require_python()
    assert "needs Python 3.9" in str(e.value)


def test_version_names_the_isa_and_the_rtl_followed():
    p = atalla_sim("--version")
    assert p.returncode == 0, p.stdout
    assert p.stdout.startswith("atalla-sim ")
    assert "ISA: " in p.stdout and " opcodes, table " in p.stdout
    assert "scheduler_integration_SP26_joshklug" in p.stdout
