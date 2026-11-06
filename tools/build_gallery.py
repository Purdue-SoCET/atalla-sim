#!/usr/bin/env python3
"""A static site of activity traces for the reference kernels: `atalla-sim gallery OUT`.

    atalla-sim gallery site/          # then host site/ (GitHub Pages, a web server)
    atalla-sim view --serve site/     # or look at it locally

writes

    site/index.html     the kernels with their cycles (against the baseline),
                        each linked to its trace in the viewer, and the version
    site/viewer.html    the activity viewer, which loads ?trace=URL
    site/traces/*.json  one trace per frozen kernel (tests/equivalence/programs),
                        and the example MLP as a workload when the functional
                        sim's kernels are there

It needs no private submodule for the kernels (their images are frozen in
the repository), so it can run in CI on every push and publish the result.
"""

import sys
if sys.version_info < (3, 9):
    sys.exit("atalla-sim needs Python 3.9 or newer; %s is %s"
             % (sys.executable, sys.version.split()[0]))

import argparse
import html
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "src"), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import activity_view                                                  # noqa: E402
import bench                                                          # noqa: E402
from atalla import config, version                                    # noqa: E402
from atalla.atalla_platform import build_atalla_platform              # noqa: E402

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>atalla-sim gallery</title>
<style>
:root {{ --bg: #f4f5f2; --surface: #fff; --ink: #1c2420; --muted: #5d6863; --line: #d7ddd9;
        --accent: #2e6f8e; --bad: #a8321e; font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #121815; --surface: #19211d; --ink: #e2e9e5; --muted: #98a59f; --line: #2c3732;
          --accent: #7fb8d4; --bad: #f08a74; color-scheme: dark; }}
}}
body {{ margin: 0; background: var(--bg); color: var(--ink); font-size: 15px; line-height: 1.45; }}
main {{ max-width: 960px; margin: 0 auto; padding: 24px 16px 48px; }}
h1 {{ font-size: 24px; margin: 0 0 4px; }}
h2 {{ font-size: 17px; margin: 28px 0 8px; }}
p {{ color: var(--muted); margin: 4px 0; }}
.card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 8px; overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ text-align: left; padding: 7px 12px; border-bottom: 1px solid var(--line); }}
th {{ color: var(--muted); font-weight: 600; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
tr:last-child td {{ border-bottom: none; }}
a {{ color: var(--accent); }}
.bad {{ color: var(--bad); font-weight: 600; }}
pre {{ background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px;
       overflow-x: auto; font-size: 12.5px; color: var(--ink); }}
</style></head><body><main>
<h1>atalla-sim gallery</h1>
<p>Which blocks of the chip work when, for each reference kernel, on {machine}.
Built {when}. Open one to step through it cycle by cycle, or <a href="viewer.html">open the viewer</a>
and drop in a trace of your own (it stays in your browser).</p>
{workload}
<h2>Reference kernels</h2>
<div class="card"><table>
<thead><tr><th>kernel</th><th>cycles</th><th>baseline</th><th>change</th><th>trace</th></tr></thead>
<tbody>
{rows}
</tbody></table></div>
<p>The kernels are the program images frozen in <code>tests/equivalence/programs</code>; names
ending in <code>.c</code> came from the C compiler. The baseline is <code>tests/perf_baseline.json</code>.</p>
<h2>This build</h2>
<pre>{version}</pre>
</main></body></html>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="atalla-sim gallery", description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path, help="the folder to write the site to")
    ap.add_argument("--no-model", action="store_true", help="leave out the example MLP workload")
    args = ap.parse_args(argv)
    out = args.out
    (out / "traces").mkdir(parents=True, exist_ok=True)
    (out / "viewer.html").write_text(activity_view.render_empty())
    cfg = config.load()
    base = {}
    if bench.BASELINE.is_file():
        base = json.loads(bench.BASELINE.read_text())["kernels"]
    rows = []
    for name in bench.kernels():
        instr, data, regs = bench.load(name)
        plat = build_atalla_platform(instr, data, init_sregs=regs, strict=False, activity=True)
        cycles = plat.run_until_done()
        trace = plat.activity.trace({"program": name, "machine": config.summary(cfg)})
        path = "traces/%s.json" % name
        (out / path).write_text(json.dumps(trace, separators=(",", ":")))
        b = base.get(name, {}).get("cycles")
        change = "" if b is None else "same" if b == cycles else "%+.2f%%" % (100.0 * (cycles - b) / b)
        cls = "" if change in ("", "same") else ' class="bad"'
        rows.append('<tr><td>%s</td><td class="num">%s</td><td class="num">%s</td><td%s>%s</td>'
                    '<td><a href="viewer.html?trace=%s">open</a></td></tr>'
                    % (html.escape(name), "{:,}".format(cycles), "{:,}".format(b) if b else "-",
                       cls, change or "new", html.escape(path)))
        print("%-34s %7d cycles" % (name, cycles))
    workload = ""
    if not args.no_model:
        workload = model_workload(out)
    (out / "index.html").write_text(PAGE.format(
        machine=html.escape(config.summary(cfg)), when=time.strftime("%Y-%m-%d %H:%M %Z"),
        rows="\n".join(rows), workload=workload, version=html.escape(version.text())))
    print("gallery written to %s (index.html, viewer.html, %d traces)"
          % (out, len(list((out / "traces").glob("*.json")))))
    return 0


def model_workload(out: Path) -> str:
    """The example MLP's operations as one workload, when the functional
    sim's kernels are there to run it; the HTML that links it, or ""."""
    from scheduler import golden
    plan = ROOT / "examples" / "model" / "mlp.plan.json"
    if not golden.HAVE_GOLDEN or not plan.is_file():
        print("(no MLP workload: needs the functional-sim submodule's kernels)")
        return ""
    import numpy as np
    from atalla import model_runner
    p = model_runner.Plan.load(plan)
    rng = np.random.default_rng(0)
    inputs = {"arg%d" % k: rng.normal(0, 0.5, size=shape).astype(np.float32)
              for k, (_, shape, _) in enumerate(p.inputs)}
    kw = config.platform_kwargs(config.load())
    kw["activity"] = True
    report = model_runner.run_model(plan, inputs, kw, None, "")
    (out / "traces" / "mlp.json").write_text(json.dumps(report["activity"], separators=(",", ":")))
    print("%-34s %7d cycles (%d operations)" % ("mlp (workload)", report["cycles"],
                                               report["counts"]["accelerator"]))
    return ('<h2>A model</h2><p>The example MLP (Linear 4&rarr;8, ReLU, Linear 8&rarr;3, BF16), '
            'its %d accelerator operations end to end on one timeline: %s cycles. '
            '<a href="viewer.html?trace=traces/mlp.json">Open the workload</a>.</p>'
            % (report["counts"]["accelerator"], "{:,}".format(report["cycles"])))


if __name__ == "__main__":
    sys.exit(main())
