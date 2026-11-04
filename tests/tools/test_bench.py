"""atalla-sim bench (tools/bench.py): the cycle baseline of the reference kernels."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import bench  # noqa: E402


def test_the_baseline_lists_every_frozen_kernel():
    base = json.loads(bench.BASELINE.read_text())["kernels"]
    assert sorted(base) == bench.kernels()


def test_the_reference_kernels_take_the_baseline_cycles():
    """The gate: a change in any kernel's cycles fails here (and in
    tests/equivalence) until tests/perf_baseline.json is updated with it."""
    assert bench.main(["--repeat", "1"]) == 0


def test_compare_flags_changes_beyond_the_tolerance():
    base = {"a": {"cycles": 100}, "b": {"cycles": 100}, "gone": {"cycles": 5}}
    now = {"a": {"cycles": 100, "cycles_per_second": 1}, "b": {"cycles": 102, "cycles_per_second": 1},
           "new": {"cycles": 7, "cycles_per_second": 1}}
    rows = {r["kernel"]: r for r in bench.compare(now, base, tolerance=0.01)}
    assert rows["a"]["ok"] and not rows["b"]["ok"] and rows["new"]["ok"] and not rows["gone"]["ok"]
    assert "+2.00%" in bench.table(list(rows.values()))
    assert bench.compare(now, base, tolerance=0.05)[1]["ok"]
