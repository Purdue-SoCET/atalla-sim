"""models/ from source: trace, schedule, compile and link the basic model,
run its ELF and check every result against PyTorch (about 30 s). Skipped
without torch or the compiler submodule."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_basic_model_builds_runs_and_matches_pytorch(tmp_path):
    pytest.importorskip("torch")
    if not (ROOT / "third_party" / "aihw-ppci-compiler" / "ppci").is_dir():
        pytest.skip("the compiler submodule isn't checked out")
    pytest.importorskip("pydot")              # the compiler's code generator imports it
    for p in (str(ROOT / "models"), str(ROOT / "models" / "atalla-graph")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import verify_model
    report = verify_model.verify("basic", tmp_path, ROOT)
    assert report["status"] == 0
    assert report["passed"], [r for r in report["results"] if not r["passed"]]
    assert len(report["results"]) == 9
