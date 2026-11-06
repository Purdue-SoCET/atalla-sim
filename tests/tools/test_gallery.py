"""atalla-sim gallery (tools/build_gallery.py): a static site of the reference kernels' traces."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import bench  # noqa: E402
import build_gallery  # noqa: E402


def test_the_gallery_has_a_trace_and_a_link_per_kernel(tmp_path):
    assert build_gallery.main([str(tmp_path), "--no-model"]) == 0
    names = bench.kernels()
    traces = sorted(p.stem for p in (tmp_path / "traces").glob("*.json"))
    assert traces == sorted(names)
    index = (tmp_path / "index.html").read_text()
    for n in names:
        assert 'viewer.html?trace=traces/%s.json' % n in index
    assert 'class="bad"' not in index                     # every kernel at its baseline
    assert "Open a trace" in (tmp_path / "viewer.html").read_text()
    t = json.loads((tmp_path / "traces" / "gemm.json").read_text())
    assert t["cycles"] > 0 and t["blocks"]
