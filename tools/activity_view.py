#!/usr/bin/env python3
"""Draw an activity trace as one self-contained HTML page.

    python tools/run_kernel.py gemm --activity gemm.html     # record and draw
    python tools/run_kernel.py gemm --activity gemm.json     # record only
    python tools/activity_view.py gemm.json -o gemm.html     # draw later

The page shows the chip's blocks in panels -- fetch, decode, scalar execute,
the vector core, the systolic array, memory -- each coloured by what it was
doing at the chosen cycle: idle (grey), active (green) or stalled (amber).
A slider and a play button move through the run; with a window of more than
one cycle, each block shows the share of those cycles it was busy. Each block
prints its busy share over the whole run. Below, a timeline has one row per
block and, on top, the performance monitor's cycle buckets; click it to jump.

The trace comes from src/atalla/activity.py. To draw a new block, add it to
the probe: the page lays out whatever blocks the trace lists, by panel.
"""

import argparse
import json
import sys
from pathlib import Path

#: Panels in drawing order; a block's "panel" field picks one.
PANELS = ["Fetch", "Decode 1", "Decode 2", "Scalar execute", "Vector core",
          "Systolic array", "Memory"]

#: Blocks the chip has but the model doesn't simulate, drawn greyed out.
NOT_MODELLED = {"Memory": ["DDR4 controller / Ramulator2: not modelled "
                           "(DRAM is a fixed latency)"]}


def render(trace: dict, title: str = "") -> str:
    """The HTML page for a trace (ActivityProbe.trace())."""
    meta = trace.get("meta", {})
    name = title or meta.get("program") or "Atalla"
    data = json.dumps(trace, separators=(",", ":")).replace("</", "<\\/")
    extra = json.dumps({"panels": PANELS, "notModelled": NOT_MODELLED})
    return (_PAGE.replace("{{TITLE}}", _escape("%s activity" % name))
                 .replace("{{DATA}}", data).replace("{{EXTRA}}", extra))


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("trace", type=Path, help="a trace written by run_kernel.py --activity")
    ap.add_argument("-o", "--output", type=Path, help="the HTML page (default: next to the trace)")
    args = ap.parse_args()
    trace = json.loads(args.trace.read_text())
    out = args.output or args.trace.with_suffix(".html")
    out.write_text(render(trace))
    print("activity view written to %s" % out)


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<style>
:root {
  --bg: #f4f5f3; --surface: #ffffff; --ink: #1c2420; --muted: #5d6863; --line: #d7ddd9;
  --idle: #e3e7e4; --active: #2f8a55; --stalled: #c98a12; --cursor: #a94f23;
  --b-issue: #2f8a55; --b-stall: #c98a12; --b-front: #4a78b5; --b-halt: #8b8f94;
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #111714; --surface: #18201c; --ink: #e2e9e5; --muted: #98a59f; --line: #2c3732;
          --idle: #26302b; --active: #4fbf7d; --stalled: #e3aa3c; --cursor: #e4905f; color-scheme: dark; }
}
body { margin: 0; background: var(--bg); color: var(--ink); font-size: 14px; }
header, main { padding: 12px 16px; max-width: 1400px; margin: 0 auto; }
h1 { font-size: 20px; margin: 4px 0; }
.sub { color: var(--muted); }
.controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 10px 0;
            position: sticky; top: 0; background: var(--bg); padding: 8px 0; z-index: 2; }
.controls input[type=range] { flex: 1 1 300px; }
button, select, input { font: inherit; color: var(--ink); background: var(--surface);
                        border: 1px solid var(--line); border-radius: 4px; padding: 3px 8px; }
#cyc { font-variant-numeric: tabular-nums; min-width: 9ch; }
.legend { display: flex; gap: 12px; color: var(--muted); }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
.chip { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; }
.panel { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 8px; min-width: 0; }
.panel.wide { grid-column: 1 / -1; }
.panel h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin: 0 0 6px; }
.blocks { display: flex; flex-wrap: wrap; gap: 5px; }
.blk { flex: 1 1 120px; border-radius: 4px; padding: 5px 7px; background: var(--idle);
       border: 1px solid transparent; min-width: 0; cursor: default; }
.blk .l { font-weight: 600; font-size: 12.5px; overflow-wrap: anywhere; }
.blk .u { font-size: 11px; opacity: .72; font-variant-numeric: tabular-nums; }
.blk.lane { flex: 0 0 auto; width: 42px; padding: 4px 2px; text-align: center; }
.blk.off { background: transparent; border: 1px dashed var(--line); color: var(--muted); }
.timeline { margin-top: 14px; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 8px; }
.timeline .bar { display: flex; gap: 8px; align-items: center; margin-bottom: 6px; color: var(--muted); }
#tl { width: 100%; display: block; cursor: crosshair; }
</style></head><body>
<header>
  <h1>{{TITLE}}</h1>
  <div class="sub" id="sub"></div>
</header>
<main>
  <div class="controls">
    <button id="play" type="button">Play</button>
    <button id="back" type="button" title="one cycle back">&#9664;</button>
    <button id="fwd" type="button" title="one cycle forward">&#9654;</button>
    <input id="slider" type="range" min="0" value="0" aria-label="cycle">
    <span id="cyc"></span>
    <label>speed <select id="speed"><option>10</option><option selected>100</option><option>1000</option><option>10000</option></select> cycles/s</label>
    <label>window <select id="win"><option value="1">1 cycle</option><option>10</option><option>100</option><option>1000</option></select></label>
    <span class="legend"><span><span class="sw" style="background:var(--idle)"></span>idle</span>
      <span><span class="sw" style="background:var(--active)"></span>active</span>
      <span><span class="sw" style="background:var(--stalled)"></span>stalled</span></span>
  </div>
  <div class="chip" id="chip"></div>
  <div class="timeline">
    <div class="bar"><span>Timeline: click to jump.</span>
      <button id="zin" type="button">Zoom in</button><button id="zout" type="button">Zoom out</button>
      <button id="zfit" type="button">Whole run</button><span id="range"></span></div>
    <canvas id="tl"></canvas>
  </div>
</main>
<script id="trace" type="application/json">{{DATA}}</script>
<script>
(function () {
  "use strict";
  var T = JSON.parse(document.getElementById("trace").textContent);
  var X = {{EXTRA}};
  var N = Math.max(1, T.cycles), cycle = 0, playing = false, last = 0, acc = 0;
  var view = [0, N - 1];
  var css = getComputedStyle(document.documentElement);
  function col(name) { return css.getPropertyValue(name).trim(); }

  // Per block: interval starts and running totals, for O(log n) lookups.
  T.blocks.forEach(function (b) {
    b.starts = b.intervals.map(function (r) { return r[0]; });
    var ca = [0], cs = [0];
    b.intervals.forEach(function (r) {
      var n = r[1] - r[0] + 1;
      ca.push(ca[ca.length - 1] + (r[2] === 1 ? n : 0));
      cs.push(cs[cs.length - 1] + (r[2] === 2 ? n : 0));
    });
    b.ca = ca; b.cs = cs;
    b.busy = (ca[ca.length - 1] + cs[cs.length - 1]) / N;
  });
  function find(starts, c) {           // last index with start <= c, or -1
    var lo = 0, hi = starts.length - 1, ans = -1;
    while (lo <= hi) { var m = (lo + hi) >> 1; if (starts[m] <= c) { ans = m; lo = m + 1; } else hi = m - 1; }
    return ans;
  }
  function stateAt(b, c) {
    var i = find(b.starts, c);
    return (i >= 0 && b.intervals[i][1] >= c) ? b.intervals[i][2] : 0;
  }
  function upto(b, c, which) {          // active (1) or stalled (2) cycles in [0, c]
    if (c < 0) return 0;
    var i = find(b.starts, c), cum = which === 1 ? b.ca : b.cs;
    if (i < 0) return 0;
    var r = b.intervals[i], n = cum[i];
    if (r[2] === which) n += Math.min(c, r[1]) - r[0] + 1;
    return n;
  }

  // The chip.
  var chip = document.getElementById("chip"), els = {};
  var panels = X.panels.slice();
  T.blocks.forEach(function (b) { if (panels.indexOf(b.panel) < 0) panels.push(b.panel); });
  panels.forEach(function (p) {
    var mine = T.blocks.filter(function (b) { return b.panel === p; });
    if (!mine.length) return;
    var div = document.createElement("section");
    div.className = "panel" + (p === "Memory" || p === "Vector core" ? " wide" : "");
    div.innerHTML = "<h2></h2><div class='blocks'></div>";
    div.querySelector("h2").textContent = p;
    var box = div.querySelector(".blocks");
    mine.forEach(function (b) {
      var e = document.createElement("div");
      var lane = /^lane\d+$/.test(b.id);
      e.className = "blk" + (lane ? " lane" : "");
      e.id = "blk-" + b.id;
      e.innerHTML = "<div class='l'></div><div class='u'></div>";
      e.querySelector(".l").textContent = lane ? b.id.slice(4) : b.label;
      e.querySelector(".u").textContent = Math.round(100 * b.busy) + "%";
      var tot = b.ca[b.ca.length - 1], st = b.cs[b.cs.length - 1];
      e.title = b.label + "\nactive: " + (b.active || "never") + (b.stalled ? "\nstalled: " + b.stalled : "") +
                "\n\n" + tot + " cycles active, " + st + " stalled (" + Math.round(100 * b.busy) + "% of the run)";
      box.appendChild(e);
      els[b.id] = e;
    });
    (X.notModelled[p] || []).forEach(function (t) {
      var e = document.createElement("div");
      e.className = "blk off"; e.innerHTML = "<div class='l'></div>";
      e.querySelector(".l").textContent = t;
      box.appendChild(e);
    });
    chip.appendChild(div);
  });

  function mix(c1, frac) {               // c1 at `frac` over the idle colour
    var a = hex(col("--idle")), b = hex(c1);
    return "rgb(" + [0, 1, 2].map(function (i) { return Math.round(a[i] + (b[i] - a[i]) * frac); }).join(",") + ")";
  }
  function hex(h) {
    h = h.replace("#", "");
    return [0, 2, 4].map(function (i) { return parseInt(h.substr(i, 2), 16); });
  }

  // Drawing the chip at the current cycle.
  var win = 1;
  function paint() {
    document.getElementById("cyc").textContent = "cycle " + cycle.toLocaleString();
    document.getElementById("slider").value = cycle;
    T.blocks.forEach(function (b) {
      var e = els[b.id];
      if (win <= 1) {
        var s = stateAt(b, cycle);
        e.style.background = s === 1 ? col("--active") : s === 2 ? col("--stalled") : "";
        e.style.color = s ? "#fff" : "";
      } else {
        var lo = Math.max(0, cycle - win + 1), n = cycle - lo + 1;
        var fa = (upto(b, cycle, 1) - upto(b, lo - 1, 1)) / n;
        var fs = (upto(b, cycle, 2) - upto(b, lo - 1, 2)) / n;
        e.style.background = (fa || fs) ? mix(fa >= fs ? col("--active") : col("--stalled"), Math.max(fa, fs)) : "";
        e.style.color = Math.max(fa, fs) > 0.5 ? "#fff" : "";
      }
    });
    drawTimeline();
  }

  // The timeline.
  var cv = document.getElementById("tl"), ctx = cv.getContext("2d");
  var LABEL = 160, ROW = 13, TOP = 18;
  var BUCKET = function (name) {
    return name === "issue" ? col("--b-issue") : name.indexOf("stall") === 0 ? col("--b-stall")
         : name.indexOf("frontend") === 0 ? col("--b-front") : col("--b-halt");
  };
  function drawTimeline() {
    var w = cv.clientWidth, h = TOP + ROW * T.blocks.length + 4, dpr = window.devicePixelRatio || 1;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); cv.style.height = h + "px";
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    var span = view[1] - view[0] + 1, pw = Math.max(1, w - LABEL);
    function x(c) { return LABEL + (c - view[0]) * pw / span; }
    ctx.font = "10px sans-serif"; ctx.textBaseline = "middle";
    ctx.fillStyle = col("--muted"); ctx.fillText("cycle buckets", 4, TOP / 2);
    (T.buckets || []).forEach(function (r) {
      if (r[1] < view[0] || r[0] > view[1]) return;
      ctx.fillStyle = BUCKET(r[2]);
      ctx.fillRect(x(r[0]), 2, Math.max(1, x(r[1] + 1) - x(r[0])), TOP - 5);
    });
    T.blocks.forEach(function (b, k) {
      var y = TOP + k * ROW;
      ctx.fillStyle = col("--muted"); ctx.fillText(b.label, 4, y + ROW / 2);
      ctx.fillStyle = col("--idle"); ctx.fillRect(LABEL, y + 1, pw, ROW - 2);
      b.intervals.forEach(function (r) {
        if (r[1] < view[0] || r[0] > view[1]) return;
        ctx.fillStyle = r[2] === 1 ? col("--active") : col("--stalled");
        ctx.fillRect(x(r[0]), y + 1, Math.max(1, x(r[1] + 1) - x(r[0])), ROW - 2);
      });
    });
    if (cycle >= view[0] && cycle <= view[1]) {
      ctx.fillStyle = col("--cursor"); ctx.fillRect(x(cycle), 0, 2, h);
    }
    document.getElementById("range").textContent =
      "cycles " + view[0].toLocaleString() + " to " + view[1].toLocaleString();
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

  // Controls.
  var slider = document.getElementById("slider");
  slider.max = N - 1;
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
        var span = view[1] - view[0] + 1; view = [Math.min(cycle, N - span), Math.min(cycle, N - span) + span - 1];
      }
      paint();
      if (cycle >= N - 1) { playing = false; playBtn.textContent = "Play"; return; }
    }
    requestAnimationFrame(step);
  }
  window.addEventListener("resize", drawTimeline);

  var m = T.meta || {}, bits = [N.toLocaleString() + " cycles"];
  ["program", "array", "lanes", "burst_bytes", "dram_latency"].forEach(function (k) {
    if (m[k] !== undefined && k !== "program") bits.push(k.replace("_", " ") + " " + m[k]);
  });
  document.getElementById("sub").textContent = bits.join(" · ") +
    ". Hover a block for what its states mean; the percentage is its busy share of the run.";
  paint();
})();
</script>
</body></html>
"""


if __name__ == "__main__":
    sys.exit(main())
