"""The activity viewer page (tools/activity_view.py): traces, workloads, the
empty hosted viewer, and serving a folder."""
import gzip
import json
import re
import shutil
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import activity_view  # noqa: E402
from atalla.atalla_platform import build_atalla_platform  # noqa: E402
from scheduler.core import load_program_text  # noqa: E402

CHROME = shutil.which("chromium-browser") or shutil.which("chromium") or shutil.which("google-chrome")


def trace(name):
    instr, data = load_program_text(gzip.decompress(
        (ROOT / ("tests/equivalence/programs/%s.in.gz" % name)).read_bytes()).decode())
    plat = build_atalla_platform(instr, data, activity=True, strict=False)
    plat.run_until_done()
    return plat.activity.trace({"program": name})


@pytest.fixture(scope="module")
def traces():
    return {n: trace(n) for n in ("relu", "add")}


def embedded(page):
    return json.loads(re.search(r'<script id="trace" type="application/json">(.*?)</script>',
                                page, re.S).group(1))


def test_a_workload_embeds_its_layers_in_order(traces):
    w = activity_view.workload("demo", [("relu", traces["relu"]), ("add", traces["add"])])
    page = activity_view.render(w)
    assert [L["name"] for L in embedded(page)["layers"]] == ["relu", "add"]
    assert "<title>demo activity</title>" in page


def test_the_empty_viewer_has_no_trace_and_says_how_to_get_one():
    page = activity_view.render_empty()
    assert embedded(page) is None
    assert "Open a trace" in page and "?trace=URL" in page


def test_join_on_the_command_line(traces, tmp_path):
    for n, t in traces.items():
        (tmp_path / ("%s.json" % n)).write_text(json.dumps(t))
    assert activity_view.main([str(tmp_path / "relu.json"), str(tmp_path / "add.json"),
                               "--join", "pair", "-o", str(tmp_path / "pair.json")]) == 0
    doc = json.loads((tmp_path / "pair.json").read_text())
    assert doc["workload"] == "pair" and [L["name"] for L in doc["layers"]] == ["relu", "add"]


def test_serve_lists_traces_and_serves_the_viewer(traces, tmp_path):
    (tmp_path / "relu.json").write_text(json.dumps(traces["relu"]))
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    t = threading.Thread(target=activity_view.serve, args=(tmp_path, port), daemon=True)
    t.start()
    import time
    for _ in range(50):
        try:
            index = urllib.request.urlopen("http://127.0.0.1:%d/" % port, timeout=2).read().decode()
            break
        except OSError:
            time.sleep(0.1)
    assert 'viewer.html?trace=relu.json' in index
    viewer = urllib.request.urlopen("http://127.0.0.1:%d/viewer.html" % port).read().decode()
    assert "Open a trace" in viewer
    got = json.loads(urllib.request.urlopen("http://127.0.0.1:%d/relu.json" % port).read())
    assert got["cycles"] == traces["relu"]["cycles"]


@pytest.mark.skipif(CHROME is None, reason="no headless Chromium to run the page's script")
@pytest.mark.parametrize("kind", ["trace", "workload"])
def test_the_page_script_runs_in_a_browser(traces, tmp_path, kind):
    doc = (traces["relu"] if kind == "trace" else
           activity_view.workload("demo", [("relu", traces["relu"]), ("add", traces["add"])]))
    page = tmp_path / "p.html"
    page.write_text(activity_view.render(doc))
    p = subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-sandbox", "--enable-logging=stderr",
                        "--v=0", "--dump-dom", page.as_uri()], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, universal_newlines=True, timeout=120)
    assert "Uncaught" not in p.stderr, p.stderr[-2000:]
    dom = p.stdout
    assert '<main id="view">' in dom                       # shown, not hidden
    assert re.search(r'<span id="cyc">cycle 0</span>', dom)
    rows = dom.count('<tr tabindex="0"')
    assert rows == (2 if kind == "workload" else 0)
