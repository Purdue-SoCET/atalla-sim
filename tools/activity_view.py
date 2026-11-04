#!/usr/bin/env python3
"""The activity viewer: which blocks of the chip work when, as one HTML page.

    atalla-sim run gemm --activity gemm.html          # record and draw
    atalla-sim run gemm --activity gemm.json          # record only
    atalla-sim view gemm.json -o gemm.html            # draw later
    atalla-sim view --join mlp a.json b.json c.json -o mlp.json   # a workload: traces end to end
    atalla-sim view --empty -o viewer.html            # the viewer alone, for hosting
    atalla-sim view --serve traces/                   # serve a folder of traces on localhost

The page draws the chip as the model builds it -- Fetch, Decode1, Decode2,
scalar execute, the vector core, the systolic array and memory -- and colours
every block by what it was doing at the chosen cycle: idle, active (moving
work) or stalled (holding work it can't move). A slider and a play button
move through the run; with a window of more than one cycle, each block is
shaded by how much of that window it was busy. A block's percentage is the
share of the run's cycles in which it had work (active or stalled). Below,
a timeline has one row per block and, on top, the performance monitor's cycle
buckets; click it to jump there.

One page shows any trace: the one embedded in it, one the viewer opens or
drops on it (read in the browser, never uploaded), or one at `?trace=URL`
when the page is hosted. A **workload** is several runs, one per layer:

    {"version": 1, "workload": "mlp", "meta": {...},
     "layers": [{"name": "matmul 0", "trace": <a trace>}, ...]}

The page joins them end to end on one timeline, marks where each layer
starts, and lists the layers with their cycles and busiest blocks; clicking
one zooms to it.

The trace comes from src/atalla/activity.py. The drawing is the LAYOUT
function in the page's script: one entry per box, with the block ids it
shows. A block the layout doesn't place is listed under the drawing, so a
new probe block appears without changes here; give it a box to draw it in
place.
"""

import sys
if sys.version_info < (3, 9):
    sys.exit("atalla-sim needs Python 3.9 or newer; %s is %s"
             % (sys.executable, sys.version.split()[0]))

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple


def _data(obj) -> str:
    return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render(trace: dict, title: str = "") -> str:
    """The page with `trace` (ActivityProbe.trace(), or a workload) in it."""
    meta = trace.get("meta", {})
    name = title or trace.get("workload") or meta.get("program") or "Atalla"
    return _PAGE.replace("{{TITLE}}", _escape("%s activity" % name)).replace("{{DATA}}", _data(trace))


def render_empty(title: str = "Atalla activity viewer") -> str:
    """The viewer with no trace: it opens files, drops and ?trace=URL."""
    return _PAGE.replace("{{TITLE}}", _escape(title)).replace("{{DATA}}", "null")


def workload(name: str, layers: Iterable[Tuple[str, dict]], meta: Optional[dict] = None) -> dict:
    """Several traces, in the order they ran, as one workload."""
    return {"version": 1, "workload": name, "meta": dict(meta or {}),
            "layers": [{"name": n, "trace": t} for n, t in layers]}


def serve(folder: Path, port: int = 8000, host: str = "127.0.0.1") -> None:
    """Serve `folder` with the viewer at /viewer.html and an index of its traces at /."""
    import html
    import http.server
    import urllib.parse

    folder = folder.resolve()
    viewer = render_empty().encode()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(folder), **kw)

        def _send(self, body: bytes, kind: str = "text/html; charset=utf-8"):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path
            if path == "/viewer.html":
                return self._send(viewer)
            if path == "/":
                rows = "".join('<li><a href="viewer.html?trace=%s">%s</a></li>'
                               % (urllib.parse.quote(str(p.relative_to(folder))),
                                  html.escape(str(p.relative_to(folder))))
                               for p in sorted(folder.rglob("*.json")))
                return self._send(("<!doctype html><meta charset=utf-8><title>Traces</title>"
                                   "<h1>Activity traces in %s</h1><ul>%s</ul>"
                                   % (html.escape(str(folder)), rows or "<li>none</li>")).encode())
            return super().do_GET()

    with http.server.ThreadingHTTPServer((host, port), Handler) as httpd:
        print("serving %s at http://%s:%d/ (Ctrl-C stops)" % (folder, host, port))
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="atalla-sim view", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n")[1])
    ap.add_argument("traces", nargs="*", type=Path,
                    help="traces written by run_kernel.py --activity (or a workload)")
    ap.add_argument("-o", "--output", type=Path,
                    help="the page (.html) or, with --join, the workload (.json or .html)")
    ap.add_argument("--join", metavar="NAME",
                    help="join the traces, in order, into one workload called NAME")
    ap.add_argument("--empty", action="store_true", help="write the viewer with no trace in it")
    ap.add_argument("--serve", metavar="DIR", type=Path,
                    help="serve DIR's traces and the viewer on localhost")
    ap.add_argument("--port", type=int, default=8000, help="with --serve (default 8000)")
    args = ap.parse_args(argv)
    if args.serve:
        serve(args.serve, args.port)
        return 0
    if args.empty:
        out = args.output or Path("viewer.html")
        out.write_text(render_empty())
        print("viewer written to %s" % out)
        return 0
    if not args.traces:
        ap.error("give a trace, or --empty, or --serve DIR")
    loaded = [json.loads(p.read_text()) for p in args.traces]
    if args.join:
        doc = workload(args.join, [(p.stem, t) for p, t in zip(args.traces, loaded)])
        out = args.output or Path("%s.json" % args.join)
        out.write_text(render(doc) if out.suffix == ".html" else json.dumps(doc, separators=(",", ":")))
        print("workload of %d traces written to %s" % (len(loaded), out))
        return 0
    if len(loaded) != 1:
        ap.error("one trace at a time, or --join NAME to make a workload of several")
    out = args.output or args.traces[0].with_suffix(".html")
    out.write_text(render(loaded[0]))
    print("activity view written to %s" % out)
    return 0


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<style>
:root {
  --bg: #f2f3f0; --surface: #ffffff; --ink: #1c2420; --muted: #5d6863; --line: #d7ddd9;
  --idle-row: #e6eae7; --active: #2e8b57; --stalled: #d18f12; --cursor: #a94f23;
  --b-issue: #2e8b57; --b-stall: #d18f12; --b-front: #4a78b5; --b-halt: #8b8f94;
  --layer-a: #dfe6ef; --layer-b: #eef1f5; --err: #a8321e;
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #121815; --surface: #19211d; --ink: #e2e9e5; --muted: #98a59f; --line: #2c3732;
          --idle-row: #26302b; --cursor: #e4905f; --layer-a: #26313d; --layer-b: #1e2731;
          --err: #f08a74; color-scheme: dark; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font-size: 14px; }
header, main { padding: 12px 16px; max-width: 1240px; margin: 0 auto; }
h1 { font-size: 21px; margin: 4px 0 2px; }
.sub { color: var(--muted); margin: 0; }
.controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; padding: 8px 0;
            position: sticky; top: 0; background: var(--bg); z-index: 2; }
.controls input[type=range] { flex: 1 1 280px; }
button, select { font: inherit; color: var(--ink); background: var(--surface);
                 border: 1px solid var(--line); border-radius: 4px; padding: 3px 9px; }
button:focus-visible, select:focus-visible, input:focus-visible { outline: 2px solid var(--cursor); outline-offset: 2px; }
#cyc { font-variant-numeric: tabular-nums; min-width: 10ch; font-weight: 600; }
.legend { display: flex; flex-wrap: wrap; gap: 12px; color: var(--muted); align-items: center; }
.sw { display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; border: 1px solid #0002; }
.card { background: #fbfbf8; border: 1px solid var(--line); border-radius: 8px; padding: 8px; overflow-x: auto; }
#chip { display: block; width: 100%; min-width: 760px; height: auto; }
#chip text { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; pointer-events: none; }
#chip .box { cursor: default; }
.note { color: var(--muted); font-size: 12.5px; margin: 6px 2px 0; }
.extra { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.extra span { border: 1px solid var(--line); border-radius: 4px; padding: 2px 6px; font-size: 12px; background: var(--surface); }
.timeline { margin-top: 14px; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 8px; }
.timeline .bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 6px; color: var(--muted); }
#tl { width: 100%; display: block; cursor: crosshair; }
.load { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 6px; color: var(--muted); }
.load label { font: inherit; color: var(--ink); background: var(--surface); border: 1px solid var(--line);
              border-radius: 4px; padding: 3px 9px; cursor: pointer; }
.load label:focus-within { outline: 2px solid var(--cursor); outline-offset: 2px; }
#err { color: var(--err); }
#empty { max-width: 760px; margin: 40px auto; padding: 0 16px; color: var(--muted); line-height: 1.5; }
#empty code { background: var(--surface); border: 1px solid var(--line); border-radius: 3px; padding: 1px 4px; }
body.drop { outline: 3px dashed var(--cursor); outline-offset: -8px; }
#layers { margin-top: 14px; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 8px; overflow-x: auto; }
#layers table { border-collapse: collapse; width: 100%; font-size: 13px; }
#layers th, #layers td { text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--line); }
#layers td.num { text-align: right; font-variant-numeric: tabular-nums; }
#layers tbody tr { cursor: pointer; }
#layers tbody tr:hover, #layers tbody tr:focus { background: var(--idle-row); outline: none; }
</style></head><body>
<header>
  <h1 id="title">{{TITLE}}</h1>
  <p class="sub" id="sub"></p>
  <div class="load">
    <label>Open a trace&hellip; <input id="file" type="file" accept=".json,application/json" hidden></label>
    <span>or drop one on the page</span>
    <span id="err" hidden></span>
  </div>
</header>
<section id="empty">
  <p>No trace loaded. Open or drop an activity trace (<code>.json</code>) to see which
  blocks of the chip work when.</p>
  <p>Record one with <code>atalla-sim run gemm --activity gemm.json</code>, or a workload of
  several programs with <code>atalla-sim view --join</code> or <code>atalla-sim model --activity</code>.
  A hosted copy of this page also reads <code>?trace=URL</code>. Traces you open stay in
  your browser; nothing is uploaded.</p>
</section>
<main id="view" hidden>
  <div class="controls">
    <button id="play" type="button">Play</button>
    <button id="back" type="button" aria-label="one cycle back">&#9664;</button>
    <button id="fwd" type="button" aria-label="one cycle forward">&#9654;</button>
    <input id="slider" type="range" min="0" value="0" aria-label="cycle">
    <span id="cyc"></span>
    <label>speed <select id="speed"><option>10</option><option selected>100</option><option>1000</option><option>10000</option></select> cycles/s</label>
    <label>window <select id="win"><option value="1">1 cycle</option><option>10</option><option>100</option><option>1000</option></select></label>
  </div>
  <div class="legend">
    <span><span class="sw" style="background:#ffffff"></span>idle</span>
    <span><span class="sw" style="background:var(--active)"></span>active: moving work</span>
    <span><span class="sw" style="background:var(--stalled)"></span>stalled: holding work it can't move</span>
    <span id="pctnote"></span>
  </div>
  <div class="card"><svg id="chip" viewBox="0 26 1120 642" role="img" aria-label="the chip's blocks, coloured by state"></svg></div>
  <p class="note" id="diff"></p>
  <div class="extra" id="extra"></div>
  <div class="timeline">
    <div class="bar"><span>Timeline: one row per block; click to jump.</span>
      <button id="zin" type="button">Zoom in</button><button id="zout" type="button">Zoom out</button>
      <button id="zfit" type="button">Whole run</button><span id="range"></span></div>
    <canvas id="tl"></canvas>
  </div>
  <div id="layers" hidden>
    <div class="bar">Layers, in the order they ran; click one to show it.</div>
    <table><thead><tr><th>layer</th><th>cycles</th><th>share</th><th>busiest blocks</th></tr></thead><tbody></tbody></table>
  </div>
</main>
<script id="trace" type="application/json">{{DATA}}</script>
<script>
(function () {
  "use strict";
  // The trace on show, and the view's state. show() replaces them all.
  var T = null, N = 1, cycle = 0, playing = false, last = 0, acc = 0, win = 1, view = [0, 0];
  var byId = {}, PANEL = [], BOX = [], ARROWS = [], placed = {}, unplaced = [];
  var paint = function () {};
  var ACT = "#2e8b57", STL = "#d18f12";
  var NS = "http://www.w3.org/2000/svg", svg = document.getElementById("chip");
  var css = getComputedStyle(document.documentElement);
  function col(name) { return css.getPropertyValue(name).trim(); }
  function count(prefix) {
    var n = 0;
    while (byId[prefix + n]) n++;
    return n;
  }

  // -- a workload: one trace per layer, joined end to end ------------------------------
  function join(w) {
    var blocks = [], ids = {}, buckets = [], layers = [], off = 0;
    (w.layers || []).forEach(function (L) {
      var t = L.trace, n = Math.max(1, t.cycles);
      t.blocks.forEach(function (b) {
        if (!ids[b.id]) {
          ids[b.id] = {id: b.id, label: b.label, panel: b.panel, active: b.active,
                       stalled: b.stalled, intervals: []};
          blocks.push(ids[b.id]);
        }
        b.intervals.forEach(function (r) { ids[b.id].intervals.push([r[0] + off, r[1] + off, r[2]]); });
      });
      (t.buckets || []).forEach(function (r) { buckets.push([r[0] + off, r[1] + off, r[2]]); });
      layers.push({name: L.name, start: off, end: off + n - 1, cycles: n, trace: t});
      off += n;
    });
    var first = (w.layers && w.layers[0] && w.layers[0].trace.meta) || {};
    return {version: 1, cycles: off, blocks: blocks, buckets: buckets, layers: layers,
            meta: Object.assign({}, first, w.meta || {}, {program: w.workload || first.program})};
  }

  // -- lookups: interval starts and running totals per block ---------------------------
  function index(trace) {
    trace.blocks.forEach(function (b) {
      b.starts = b.intervals.map(function (r) { return r[0]; });
      var ca = [0], cs = [0];
      b.intervals.forEach(function (r) {
        var n = r[1] - r[0] + 1;
        ca.push(ca[ca.length - 1] + (r[2] === 1 ? n : 0));
        cs.push(cs[cs.length - 1] + (r[2] === 2 ? n : 0));
      });
      b.ca = ca; b.cs = cs;
      b.act = ca[ca.length - 1]; b.stl = cs[cs.length - 1];
    });
  }
  function find(starts, c) {
    var lo = 0, hi = starts.length - 1, ans = -1;
    while (lo <= hi) { var m = (lo + hi) >> 1; if (starts[m] <= c) { ans = m; lo = m + 1; } else hi = m - 1; }
    return ans;
  }
  function stateAt(b, c) {
    var i = find(b.starts, c);
    return (i >= 0 && b.intervals[i][1] >= c) ? b.intervals[i][2] : 0;
  }
  function upto(b, c, which) {
    if (c < 0) return 0;
    var i = find(b.starts, c), cum = which === 1 ? b.ca : b.cs;
    if (i < 0) return 0;
    var r = b.intervals[i], n = cum[i];
    if (r[2] === which) n += Math.min(c, r[1]) - r[0] + 1;
    return n;
  }

  // -- the layout ----------------------------------------------------------------------
  // Panels: [title, x, y, w, h, fill, border]. Boxes: {ids, x, y, w, h, label, kind}
  // kind "unit" (white) or "store" (blue); "off": drawn dashed, not modelled.
  function panel(t, x, y, w, h, fill, edge) { PANEL.push([t, x, y, w, h, fill, edge]); }
  function box(ids, x, y, w, h, label, kind, extra) {
    ids = [].concat(ids || []).filter(function (i) { return byId[i]; });
    ids.forEach(function (i) { placed[i] = true; });
    BOX.push(Object.assign({ids: ids, x: x, y: y, w: w, h: h, label: label, kind: kind || "unit"}, extra || {}));
  }
  function LAYOUT() {
    var i, k;
    // Front end.
    panel("Fetch", 10, 34, 122, 262, "#fde6d3", "#efc6a3");
    box("bpu", 20, 62, 102, 62, "Branch\nPrediction\nUnit");
    box("pc", 20, 134, 102, 28, "PC");
    box("icache", 20, 172, 102, 112, "iCache", "store");
    panel("Decode1", 142, 34, 150, 262, "#fff2c9", "#ecd48a");
    for (i = 0; i < 4; i++) box("dec1_s" + i, 152, 62 + i * 56, 58, 46, "Decode\n" + i);
    box("decode1_xbar", 220, 62, 62, 214, "xbar", "unit", {xbar: true});
    panel("Decode2", 302, 34, 330, 262, "#f9d6d6", "#e7aeae");
    ["ALU", "Vector", "SDMA"].forEach(function (name, c) {
      var kind = ["s", "v", "d"][c];
      for (i = 0; i < 4; i++) box("d2_" + kind + i, 312 + c * 62, 76 + i * 30, 56, 26, name + " " + i);
    });
    box("mrf", 312, 200, 180, 26, "Mask reg file", "store");
    [["srf", "Scalar reg file", 58], ["vrf", "Vector reg file", 176]].forEach(function (rf) {
      var n = count(rf[0] + "_b"), y = rf[2];
      box(rf[0] + "_read", 502, y, 120, 15, "issue reg read", "unit", {small: true});
      for (k = 0; k < n; k++) box(rf[0] + "_b" + k, 502, y + 19 + k * 18, 120, 15,
                                  rf[1].split(" ")[0] + " RF bank " + k, "store", {small: true});
      box(rf[0] + "_oc", 502, y + 22 + n * 18, 120, 15, "operand collectors", "unit", {small: true});
    });
    // Scalar execute.
    panel("Scalar execute", 642, 34, 228, 262, "#e6dcf0", "#c8b6dd");
    box("ex1_alu", 652, 62, 100, 48, "Sclr ALU\n(EX1)");
    box("ex1_br", 760, 62, 100, 48, "Branch\n(EX1)");
    box("ex2_div", 652, 118, 100, 48, "Div / mod / sqrt\n(EX2)");
    box("ex2_cvt", 760, 118, 100, 48, "int <-> bf16\n(EX2)");
    box("ex3", 652, 174, 100, 48, "FP ALU\n(EX3)");
    box("ex4", 760, 174, 100, 48, "Multiply\n(EX4)");
    box("lsu", 652, 230, 100, 48, "SCLR LD/ST\n(EX5)");
    box("mts", 760, 230, 100, 48, "Move to\nscalar");
    // Vector core.
    var lanes = count("lane"), vls = count("vlsu");
    panel("Vector core: " + lanes + " lanes x " + (32 / Math.max(1, lanes)) + " elements", 10, 306, 860, 116, "#dbe5ec", "#a9bccb");
    var lw = Math.min(30, (520 - 20) / Math.max(1, lanes) - 3);
    for (i = 0; i < lanes; i++) {
      var per = 32 / lanes;
      box("lane" + i, 20 + i * (lw + 3), 336, lw, 74, "L" + i, "unit",
          {sub: lanes <= 16 ? (per > 1 ? (i * per) + "-" + (i * per + per - 1) : "" + i) : ""});
    }
    box("gsau", 532, 336, 74, 74, "GSAU");
    box("transpose", 612, 336, 74, 74, "Transpose");
    var vw = (860 - 692 - 4 * (vls - 1)) / Math.max(1, vls);
    for (i = 0; i < vls; i++) box("vlsu" + i, 692 + i * (vw + 4), 336, vw, 74, "Vec\nLD/ST\n" + i);
    // Systolic array.
    var meissa = !byId.psum_in;
    panel("Systolic array", 880, 34, 230, 388, "#fbd3a7", "#e3a464");
    if (meissa) box(null, 930, 60, 170, 26, "no psum input (MEISSA)", "unit", {off: true});
    else box("psum_in", 930, 60, 170, 26, "psum input fifo", "store");
    box("array_in", 890, 96, 32, 278, "input fifo", "store", {vertical: true});
    box("array", 930, 96, 170, 278, (byId.array || {label: "PE grid"}).label, "unit", {grid: true});
    box("array_out", 930, 384, 170, 28, "psum output fifo", "store");
    // Memory.
    var pads = count("spad"), dmas = count("sdma");
    panel("Memory", 10, 432, 1100, 230, "#dcefb5", "#b2d26f");
    box("dcache", 20, 462, 150, 96, "dCache", "store");
    box("icache_port", 20, 566, 72, 34, "iCache\nport", "unit", {small: true});
    box("dcache_port", 98, 566, 72, 34, "dCache\nport", "unit", {small: true});
    var pw = (760 - 6 * (pads - 1)) / Math.max(1, pads);
    for (i = 0; i < dmas; i++) box("sdma" + i, 180 + i * (pw + 6), 462, pw, 24, "DMA " + i, "unit", {small: true});
    for (i = 0; i < pads; i++) {
      var x = 180 + i * (pw + 6);
      box("spad" + i + "_w", x, 492, pw, 16, "xbar (writes)", "unit", {small: true});
      box("spad" + i, x, 512, pw, 52, "Scratchpad " + i + "\n32 banks, 1RW", "store");
      box("spad" + i + "_r", x, 568, pw, 16, "xbar (reads)", "unit", {small: true});
    }
    box("dram", 950, 462, 150, 122, /Ramulator/.test((byId.dram || {}).label || "")
        ? "DRAM\nRamulator 2" : "DRAM\nfixed latency", "store");
    if (byId.axi_r) {
      box("axi_r", 20, 612, 537, 26, "AXI read path: a burst taken a cycle, beats back one a cycle", "unit");
      box("axi_w", 563, 612, 537, 26, "AXI write path: one burst at a time, a beat a cycle", "unit");
    } else {
      box("channel", 20, 612, 1080, 26, "Shared DRAM channel: one request a cycle, shared by the caches and the DMAs", "unit");
    }
    // Data flow.
    ARROWS.push([132, 165, 142, 165], [292, 165, 302, 165], [632, 165, 642, 165],
                [466, 296, 466, 306], [870, 372, 888, 372],
                [775, 422, 775, 490], [1025, 584, 1025, 610], [56, 600, 56, 610]);
  }

  function drawChip() {
  svg.textContent = "";
  // -- drawing the chip ----------------------------------------------------------------
    function el(tag, attrs, parent) {
      var e = document.createElementNS(NS, tag);
      for (var a in attrs) e.setAttribute(a, attrs[a]);
      (parent || svg).appendChild(e);
      return e;
    }
    function text(x, y, s, attrs, parent) {
      var lines = String(s).split("\n"), size = +(attrs && attrs["font-size"] || 12);
      var t = el("text", Object.assign({x: x, y: y - (lines.length - 1) * size * 0.6, "text-anchor": "middle",
                                        "dominant-baseline": "middle", fill: "#1d2421", "font-size": size}, attrs || {}), parent);
      lines.forEach(function (line, n) {
        var span = el("tspan", {x: x, dy: n ? size * 1.2 : 0}, t);
        span.textContent = line;
      });
      return t;
    }
    var defs = el("defs", {});
    var mk = el("marker", {id: "ah", viewBox: "0 0 8 8", refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: "auto"}, defs);
    el("path", {d: "M0,0 L8,4 L0,8 z", fill: "#7c8580"}, mk);
    PANEL.forEach(function (p) {
      el("rect", {x: p[1], y: p[2], width: p[3], height: p[4], rx: 8, fill: p[5], stroke: p[6]});
      text(p[1] + 10, p[2] + 14, p[0].toUpperCase(), {"text-anchor": "start", "font-size": 11,
           "font-weight": 700, "letter-spacing": "0.06em", fill: "#3b4440"});
    });
    ARROWS.forEach(function (a) {
      el("line", {x1: a[0], y1: a[1], x2: a[2], y2: a[3], stroke: "#7c8580", "stroke-width": 1.5, "marker-end": "url(#ah)"});
    });
    var BASE = {unit: "#ffffff", store: "#d8e8f7"};
    BOX.forEach(function (bx) {
      var g = el("g", {"class": "box"});
      bx.rect = el("rect", {x: bx.x, y: bx.y, width: bx.w, height: bx.h, rx: 5, fill: bx.off ? "none" : BASE[bx.kind],
                            stroke: bx.off ? "#7c8580" : "#8d9a93", "stroke-dasharray": bx.off ? "4 3" : null}, g);
      if (bx.xbar) {
        el("line", {x1: bx.x + 6, y1: bx.y + 6, x2: bx.x + bx.w - 6, y2: bx.y + bx.h - 22, stroke: "#9aa39f"}, g);
        el("line", {x1: bx.x + bx.w - 6, y1: bx.y + 6, x2: bx.x + 6, y2: bx.y + bx.h - 22, stroke: "#9aa39f"}, g);
      }
      if (bx.grid) {
        bx.cells = [];
        for (var r = 0; r < 6; r++) for (var c = 0; c < 5; c++)
          bx.cells.push(el("rect", {x: bx.x + 10 + c * 31, y: bx.y + 10 + r * 38, width: 26, height: 30, rx: 3,
                                     fill: "#ffffff", stroke: "#b48a80"}, g));
      }
      var small = bx.small || bx.h < 30, size = small ? 10 : 12;
      var ly = bx.grid ? bx.y + bx.h - 14 : bx.xbar ? bx.y + bx.h - 26 : bx.y + bx.h / 2 - (bx.h >= 44 ? 6 : 0);
      if (bx.vertical) {
        bx.label_el = text(bx.x + bx.w / 2, bx.y + bx.h / 2, bx.label, {"font-size": 11,
          transform: "rotate(-90 " + (bx.x + bx.w / 2) + " " + (bx.y + bx.h / 2) + ")"}, g);
      } else {
        bx.label_el = text(bx.x + bx.w / 2, ly, bx.label, {"font-size": size, "font-weight": small ? 400 : 600}, g);
      }
      if (bx.sub) text(bx.x + bx.w / 2, bx.y + bx.h - 26, bx.sub, {"font-size": 9, fill: "#4b5550"}, g);
      if (bx.ids.length && !bx.vertical && bx.h >= 44 && bx.w >= 26) {
        bx.pct = text(bx.x + bx.w / 2, bx.y + bx.h - (bx.grid ? 30 : 9), "", {"font-size": 10, fill: "#4b5550"}, g);
      }
      var tip = el("title", {}, g), busyAll = 0;
      var lines = [bx.label.replace(/\n/g, " ")];
      if (bx.off) lines.push("Not in the model.");
      bx.ids.forEach(function (id) {
        var b = byId[id], busy = b.act + b.stl;
        busyAll = Math.max(busyAll, busy);
        if (bx.ids.length > 1) lines.push("", b.label + ":");
        lines.push("busy " + busy.toLocaleString() + " of " + N.toLocaleString() + " cycles (" + pct(busy) + "): " +
                   b.act.toLocaleString() + " active, " + b.stl.toLocaleString() + " stalled");
        if (b.active) lines.push("active = " + b.active);
        if (b.stalled) lines.push("stalled = " + b.stalled);
      });
      tip.textContent = lines.join("\n");
      if (bx.pct) bx.pct.textContent = pct(busyAll);
    });
    function pct(n) {
      var p = 100 * n / N;
      return (p > 0 && p < 1 ? "<1" : Math.round(p)) + "%";
    }

    unplaced = T.blocks.filter(function (b) { return !placed[b.id]; });
    var extra = document.getElementById("extra");
    extra.textContent = "";
    unplaced.forEach(function (b) {
      var s = document.createElement("span"); s.id = "x-" + b.id; s.textContent = b.label;
      s.title = b.label + ": active = " + (b.active || "never"); extra.appendChild(s);
    });

    function hex(h) { h = h.replace("#", ""); return [0, 2, 4].map(function (i) { return parseInt(h.substr(i, 2), 16); }); }
    function mix(a, b, f) {
      var x = hex(a), y = hex(b);
      return "rgb(" + [0, 1, 2].map(function (i) { return Math.round(x[i] + (y[i] - x[i]) * f); }).join(",") + ")";
    }
    paint = function () {
      document.getElementById("cyc").textContent = "cycle " + cycle.toLocaleString();
      document.getElementById("slider").value = cycle;
      BOX.forEach(function (bx) {
        if (!bx.ids.length || bx.off) return;
        var base = BASE[bx.kind], fill = base, on = 0;
        if (win <= 1) {
          var st = 0;
          bx.ids.forEach(function (id) { var s = stateAt(byId[id], cycle); if (s === 1 || (s === 2 && st === 0)) st = s; });
          fill = st === 1 ? ACT : st === 2 ? STL : base; on = st ? 1 : 0;
        } else {
          var lo = Math.max(0, cycle - win + 1), n = cycle - lo + 1, fa = 0, fs = 0;
          bx.ids.forEach(function (id) {
            var b = byId[id];
            fa = Math.max(fa, (upto(b, cycle, 1) - upto(b, lo - 1, 1)) / n);
            fs = Math.max(fs, (upto(b, cycle, 2) - upto(b, lo - 1, 2)) / n);
          });
          on = Math.max(fa, fs);
          fill = on ? mix(base, fa >= fs ? ACT : STL, on) : base;
        }
        var ink = on > 0.55 ? "#ffffff" : "#1d2421";
        if (bx.cells) bx.cells.forEach(function (c) { c.setAttribute("fill", fill); });
        else bx.rect.setAttribute("fill", fill);
        if (!bx.cells) {
          bx.label_el.setAttribute("fill", ink);
          if (bx.pct) bx.pct.setAttribute("fill", on > 0.55 ? "#ffffffcc" : "#4b5550");
        }
      });
      drawTimeline();
    }

  }
  // -- the timeline ----------------------------------------------------------------------
  var cv = document.getElementById("tl"), ctx = cv.getContext("2d");
  var LABEL = 170, ROW = 13, TOP = 18, BAND = 0;
  function bucketColour(name) {
    return name === "issue" ? col("--b-issue") : name.indexOf("stall") === 0 ? col("--b-stall")
         : name.indexOf("frontend") === 0 ? col("--b-front") : col("--b-halt");
  }
  function drawTimeline() {
    var w = cv.clientWidth, h = BAND + TOP + ROW * T.blocks.length + 4, dpr = window.devicePixelRatio || 1;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); cv.style.height = h + "px";
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    var span = view[1] - view[0] + 1, pw = Math.max(1, w - LABEL);
    function x(c) { return LABEL + (c - view[0]) * pw / span; }
    ctx.font = "10px sans-serif"; ctx.textBaseline = "middle"; ctx.fillStyle = col("--muted");
    ctx.fillText("cycle buckets", 4, BAND + TOP / 2);
    if (BAND) ctx.fillText("layers", 4, BAND / 2);
    T.blocks.forEach(function (b, k) { ctx.fillText(b.label, 4, BAND + TOP + k * ROW + ROW / 2); });
    ctx.save();
    ctx.beginPath(); ctx.rect(LABEL, 0, pw, h); ctx.clip();      // bars stay right of the labels
    (T.buckets || []).forEach(function (r) {
      if (r[1] < view[0] || r[0] > view[1]) return;
      ctx.fillStyle = bucketColour(r[2]);
      ctx.fillRect(x(r[0]), BAND + 2, Math.max(1, x(r[1] + 1) - x(r[0])), TOP - 5);
    });
    T.blocks.forEach(function (b, k) {
      var y = BAND + TOP + k * ROW;
      ctx.fillStyle = col("--idle-row"); ctx.fillRect(LABEL, y + 1, pw, ROW - 2);
      b.intervals.forEach(function (r) {
        if (r[1] < view[0] || r[0] > view[1]) return;
        ctx.fillStyle = r[2] === 1 ? ACT : STL;
        ctx.fillRect(x(r[0]), y + 1, Math.max(1, x(r[1] + 1) - x(r[0])), ROW - 2);
      });
    });
    (T.layers || []).forEach(function (L, k) {
      if (L.end < view[0] || L.start > view[1]) return;
      ctx.fillStyle = k % 2 ? col("--layer-b") : col("--layer-a");
      ctx.fillRect(x(L.start), 1, Math.max(1, x(L.end + 1) - x(L.start)), BAND - 3);
      ctx.fillStyle = col("--ink");
      ctx.save(); ctx.beginPath(); ctx.rect(x(L.start), 0, Math.max(1, x(L.end + 1) - x(L.start)), BAND); ctx.clip();
      ctx.fillText(L.name, x(L.start) + 3, BAND / 2); ctx.restore();
      ctx.fillStyle = col("--muted"); ctx.fillRect(x(L.start), BAND, 1, h - BAND);
    });
    if (cycle >= view[0] && cycle <= view[1]) { ctx.fillStyle = col("--cursor"); ctx.fillRect(x(cycle) - 1, 0, 2, h); }
    ctx.restore();
    document.getElementById("range").textContent = "showing cycles " + view[0].toLocaleString() +
      " to " + view[1].toLocaleString();
  }
  cv.addEventListener("click", function (ev) {
    var r = cv.getBoundingClientRect(), px = ev.clientX - r.left;
    if (px < LABEL) return;
    var span = view[1] - view[0] + 1;
    cycle = Math.min(N - 1, Math.max(0, Math.floor(view[0] + (px - LABEL) * span / (r.width - LABEL))));
    paint();
  });
  function zoom(f) {
    var span = Math.max(20, Math.min(N, Math.round((view[1] - view[0] + 1) * f)));
    var lo = Math.max(0, Math.min(N - span, cycle - Math.floor(span / 2)));
    view = [lo, lo + span - 1]; drawTimeline();
  }
  document.getElementById("zin").onclick = function () { zoom(0.5); };
  document.getElementById("zout").onclick = function () { zoom(2); };
  document.getElementById("zfit").onclick = function () { view = [0, N - 1]; drawTimeline(); };

  // -- controls ----------------------------------------------------------------------------
  var slider = document.getElementById("slider");
  slider.oninput = function () { cycle = +slider.value; paint(); };
  document.getElementById("back").onclick = function () { cycle = Math.max(0, cycle - 1); paint(); };
  document.getElementById("fwd").onclick = function () { cycle = Math.min(N - 1, cycle + 1); paint(); };
  document.getElementById("win").onchange = function (e) { win = +e.target.value; paint(); };
  var playBtn = document.getElementById("play");
  playBtn.onclick = function () {
    playing = !playing; playBtn.textContent = playing ? "Pause" : "Play";
    if (playing) { if (cycle >= N - 1) cycle = 0; last = performance.now(); acc = 0; requestAnimationFrame(step); }
  };
  function step(now) {
    if (!playing) return;
    acc += (now - last) / 1000 * +document.getElementById("speed").value; last = now;
    var n = Math.floor(acc);
    if (n > 0) {
      acc -= n; cycle = Math.min(N - 1, cycle + n);
      if (cycle < view[0] || cycle > view[1]) {
        var span = view[1] - view[0] + 1, lo = Math.min(cycle, N - span); view = [lo, lo + span - 1];
      }
      paint();
      if (cycle >= N - 1) { playing = false; playBtn.textContent = "Play"; return; }
    }
    requestAnimationFrame(step);
  }
  window.addEventListener("resize", drawTimeline);

  function describeRun() {
  var m = T.meta || {}, bits = [N.toLocaleString() + " cycles"];
    if (m.array) bits.push("array " + m.array);
    if (m.lanes) bits.push(m.lanes + " lanes");
    if (m.burst_bytes) bits.push(m.burst_bytes + "-byte DRAM requests");
    if (m.axi) bits.push("AXI bus");
    if (m.dram === "ramulator") bits.push("DRAM timed by Ramulator 2");
    else if (m.dram_latency !== undefined) bits.push("DRAM latency " + m.dram_latency);
    document.getElementById("sub").textContent = bits.join(" · ");
    document.getElementById("pctnote").textContent =
      "Percentages: the share of the run's " + N.toLocaleString() +
      " cycles in which a block had work (active or stalled). Hover a block for the split and what its states mean.";
    document.getElementById("diff").textContent =
      "Drawn as the model builds the chip: " + count("spad") + " scratchpads, one per vector load/store unit; " +
      count("lane") + " lanes; a transpose unit beside the GSAU; separate EX2 and EX4 units. " +
      (m.dram === "ramulator" ? "DRAM is timed by Ramulator 2's DDR4 controller model. "
                              : "DRAM is a fixed-latency model (no DDR4 controller). ") +
      "Dashed: not in the model." +
      (unplaced.length ? " Also recorded, not drawn:" : "");
  }

  // -- the layers table --------------------------------------------------------------------
  function layerTable() {
    var box = document.getElementById("layers"), body = box.querySelector("tbody");
    body.textContent = "";
    box.hidden = !(T.layers && T.layers.length);
    if (box.hidden) return;
    T.layers.forEach(function (L) {
      var busy = L.trace.blocks.map(function (b) {
        var n = 0;
        b.intervals.forEach(function (r) { n += r[1] - r[0] + 1; });
        return [n, b.label];
      }).sort(function (a, b) { return b[0] - a[0]; }).slice(0, 3);
      var tr = document.createElement("tr");
      [L.name, L.cycles.toLocaleString(), Math.round(100 * L.cycles / N) + "%",
       busy.map(function (x) { return x[1] + " " + Math.round(100 * x[0] / L.cycles) + "%"; }).join(", ")]
        .forEach(function (v, k) { var td = document.createElement("td"); td.textContent = v; if (k === 1 || k === 2) td.className = "num"; tr.appendChild(td); });
      tr.tabIndex = 0;
      tr.title = "show this layer";
      tr.onclick = tr.onkeydown = function (ev) {
        if (ev.type === "keydown" && ev.key !== "Enter") return;
        view = [L.start, L.end]; cycle = L.start; paint();
      };
      body.appendChild(tr);
    });
  }

  // -- showing a trace ---------------------------------------------------------------------
  function show(trace, source) {
    if (!trace || (!trace.blocks && !trace.layers)) throw new Error("not an activity trace or workload");
    T = trace.layers && !trace.blocks ? join(trace) : trace;
    index(T);
    N = Math.max(1, T.cycles); cycle = 0; win = +document.getElementById("win").value; view = [0, N - 1];
    playing = false; document.getElementById("play").textContent = "Play";
    byId = {}; T.blocks.forEach(function (b) { byId[b.id] = b; });
    PANEL = []; BOX = []; ARROWS = []; placed = {};
    BAND = T.layers && T.layers.length ? 16 : 0;
    LAYOUT();
    drawChip();
    document.getElementById("slider").max = N - 1;
    var name = (T.meta && T.meta.program) || "Atalla";
    document.title = name + " activity";
    document.getElementById("title").textContent = name + " activity";
    describeRun();
    if (source) document.getElementById("sub").textContent += " · " + source;
    layerTable();
    document.getElementById("empty").hidden = true;
    document.getElementById("view").hidden = false;
    paint();
  }
  function fail(msg) {
    var e = document.getElementById("err");
    e.textContent = msg; e.hidden = false;
  }
  function load(text, source) {
    document.getElementById("err").hidden = true;
    try { show(JSON.parse(text), source); }
    catch (ex) { fail("Couldn't show " + source + ": " + ex.message); }
  }
  document.getElementById("file").onchange = function (ev) {
    var f = ev.target.files[0];
    if (!f) return;
    f.text().then(function (t) { load(t, f.name); });
  };
  document.addEventListener("dragover", function (ev) { ev.preventDefault(); document.body.classList.add("drop"); });
  document.addEventListener("dragleave", function (ev) { if (!ev.relatedTarget) document.body.classList.remove("drop"); });
  document.addEventListener("drop", function (ev) {
    ev.preventDefault(); document.body.classList.remove("drop");
    var f = ev.dataTransfer.files[0];
    if (f) f.text().then(function (t) { load(t, f.name); });
  });

  var embedded = JSON.parse(document.getElementById("trace").textContent);
  var url = new URLSearchParams(location.search).get("trace");
  if (embedded) show(embedded);
  else if (url) {
    fetch(url).then(function (r) {
      if (!r.ok) throw new Error(r.status + " " + r.statusText);
      return r.text();
    }).then(function (t) { load(t, url); }, function (ex) { fail("Couldn't load " + url + ": " + ex.message); });
  }
})();
</script>
</body></html>
"""


if __name__ == "__main__":
    sys.exit(main())
