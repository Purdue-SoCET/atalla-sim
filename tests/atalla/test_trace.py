"""Execution traces (src/atalla/trace.py) and atalla-sim compare-rtl."""
import gzip
from pathlib import Path

from atalla import trace
from atalla.atalla_platform import build_atalla_platform
from scheduler.core import load_program_text

PROGRAMS = Path(__file__).resolve().parents[1] / "equivalence" / "programs"


def traced(tmp_path, name, **kw):
    instr, data = load_program_text(gzip.decompress((PROGRAMS / "gemm.in.gz").read_bytes()).decode())
    plat = build_atalla_platform(instr, data, strict=False, trace=True, **kw)
    plat.run_until_done()
    path = tmp_path / name
    plat.tracer.write(str(path))
    return plat, path


def test_a_trace_has_every_issue_and_halt_and_changes_nothing(tmp_path):
    plat, path = traced(tmp_path, "a.trace")
    plain = build_atalla_platform(*load_program_text(
        gzip.decompress((PROGRAMS / "gemm.in.gz").read_bytes()).decode()), strict=False)
    assert plain.run_until_done() == plat.cycle
    events = trace.read(str(path))
    assert sum(e.kind == "issue" for e in events) == plat.core.issued_count
    assert events[-1].kind == "halt"
    assert any(e.key.startswith("v") for e in events)
    vec = next(e for e in events if e.key.startswith("v"))
    assert len(vec.value.split(",")) == 32


def test_identical_traces_compare_equal(tmp_path):
    _, a = traced(tmp_path, "a.trace")
    text, status = trace.report(str(a), str(a))
    assert status == 0 and "identical" in text


def test_a_value_change_is_found_and_timing_is_aligned(tmp_path):
    _, a = traced(tmp_path, "a.trace")
    lines = a.read_text().splitlines()
    # an RTL testbench with 5 more reset cycles, and one scalar write wrong
    shifted = [lines[0]]
    changed = False
    for ln in lines[1:]:
        c, rest = ln.split(" ", 1)
        if not changed and rest.startswith("wb x") and int(c) > 200:
            rest = rest[:-1] + ("0" if rest[-1] != "0" else "1")
            changed = int(c) + 5
        shifted.append("%d %s" % (int(c) + 5, rest))
    b = tmp_path / "rtl.trace"
    b.write_text("\n".join(shifted) + "\n")
    text, status = trace.report(str(a), str(b))
    assert status == 1
    assert "shifted by -5" in text
    first = text.splitlines()[3]
    assert "write #" in first and "value" in first, text


def test_an_issue_only_trace_compares_issues_only(tmp_path):
    _, a = traced(tmp_path, "a.trace")
    b = tmp_path / "issues.trace"
    b.write_text("\n".join(ln for ln in a.read_text().splitlines()
                           if " issue " in ln or ln.startswith("#")) + "\n")
    text, status = trace.report(str(a), str(b))
    assert status == 0 and "compared: issue" in text
