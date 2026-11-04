#!/usr/bin/env python3
"""Cycle counts and simulator speed on the reference kernels: `atalla-sim bench`.

    atalla-sim bench                       # compare with tests/perf_baseline.json
    atalla-sim bench --update              # after an intended change: new baseline
    atalla-sim bench --markdown summary.md # a table for a CI job summary or a PR

The reference kernels are the frozen program images in
tests/equivalence/programs (hand-written and compiled kernels), so the bench
needs no submodule and every run sees the same instructions. For each: the
cycles on the default machine, and how fast the simulator ran it (simulated
cycles per second of wall time).

Exit status 1 when a kernel's cycles differ from the baseline by more than
--tolerance (default 0: any change), so a timing change can't land
unnoticed; update the baseline in the same commit as an intended one. Speed
is reported, never gated: it depends on the machine running the bench.
"""

import sys
if sys.version_info < (3, 9):
    sys.exit("atalla-sim needs Python 3.9 or newer; %s is %s"
             % (sys.executable, sys.version.split()[0]))

import argparse
import gzip
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "src"),):
    if p not in sys.path:
        sys.path.insert(0, p)

from atalla.atalla_platform import build_atalla_platform, c_entry_sregs   # noqa: E402
from scheduler.core import load_program_text                               # noqa: E402

PROGRAMS = ROOT / "tests" / "equivalence" / "programs"
BASELINE = ROOT / "tests" / "perf_baseline.json"


def kernels() -> List[str]:
    return sorted(p.name[:-len(".in.gz")] for p in PROGRAMS.glob("*.in.gz"))


def load(name: str):
    instr, data = load_program_text(gzip.decompress((PROGRAMS / (name + ".in.gz")).read_bytes())
                                    .decode())
    return instr, data, (c_entry_sregs(data) if name.endswith(".c") else None)


def measure(name: str, repeat: int = 1) -> Dict[str, float]:
    instr, data, regs = load(name)
    best, cycles = float("inf"), 0
    for _ in range(repeat):
        plat = build_atalla_platform(instr, data, init_sregs=regs, strict=False)
        t = time.perf_counter()
        cycles = plat.run_until_done()
        best = min(best, time.perf_counter() - t)
    return {"cycles": int(cycles), "seconds": round(best, 4),
            "cycles_per_second": round(cycles / best) if best else 0}


def run(names: List[str], repeat: int) -> Dict[str, Dict[str, float]]:
    return {n: measure(n, repeat) for n in names}


def compare(now: Dict[str, Dict], base: Dict[str, Dict], tolerance: float) -> List[dict]:
    rows = []
    for n in sorted(set(now) | set(base)):
        a, b = base.get(n), now.get(n)
        row = {"kernel": n, "base": a["cycles"] if a else None, "now": b["cycles"] if b else None,
               "speed": b["cycles_per_second"] if b else None,
               "base_speed": a.get("cycles_per_second") if a else None}
        if a and b:
            change = (b["cycles"] - a["cycles"]) / a["cycles"] if a["cycles"] else 0.0
            row["change"] = change
            row["ok"] = abs(change) <= tolerance
        else:
            row["change"] = None
            row["ok"] = b is not None and a is None       # new kernels pass; vanished ones don't
        rows.append(row)
    return rows


def table(rows: List[dict], markdown: bool = False) -> str:
    head = ["kernel", "cycles (baseline)", "cycles (now)", "change", "sim speed (cycles/s)"]
    body = []
    for r in rows:
        change = ("new" if r["base"] is None else "gone" if r["now"] is None
                  else "%+.2f%%" % (100 * r["change"]) if r["change"] else "same")
        mark = "" if r["ok"] else (" **!**" if markdown else "  <-- changed")
        body.append([r["kernel"], "%s" % (r["base"] if r["base"] is not None else "-"),
                     "%s" % (r["now"] if r["now"] is not None else "-"), change + mark,
                     "{:,}".format(int(r["speed"])) if r["speed"] else "-"])
    if markdown:
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        lines += ["| " + " | ".join(row) + " |" for row in body]
        return "\n".join(lines)
    widths = [max(len(h), *(len(row[k]) for row in body)) for k, h in enumerate(head)]
    out = ["  ".join(h.ljust(w) for h, w in zip(head, widths))]
    out += ["  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in body]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="atalla-sim bench", description=__doc__.split("\n\n")[0])
    ap.add_argument("kernels", nargs="*", help="frozen kernels to run (default all)")
    ap.add_argument("--baseline", type=Path, default=BASELINE)
    ap.add_argument("--update", action="store_true", help="write the results as the new baseline")
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="allowed relative cycle change, e.g. 0.01 for 1%% (default 0)")
    ap.add_argument("--repeat", type=int, default=1, help="runs per kernel; the fastest is kept")
    ap.add_argument("--markdown", type=Path, help="also write the table as Markdown")
    ap.add_argument("--json", type=Path, help="also write the results as JSON")
    args = ap.parse_args(argv)
    names = args.kernels or kernels()
    unknown = [n for n in names if not (PROGRAMS / (n + ".in.gz")).is_file()]
    if unknown:
        sys.exit("no frozen kernel %s (in %s: %s)" % (", ".join(unknown), PROGRAMS,
                                                       ", ".join(kernels())))
    now = run(names, args.repeat)
    if args.json:
        args.json.write_text(json.dumps(now, indent=1))
    if args.update:
        doc = {"note": "cycles of tests/equivalence/programs on the default machine; "
                       "tools/bench.py --update rewrites it",
               "kernels": {n: {"cycles": r["cycles"]} for n, r in now.items()}}
        args.baseline.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print("baseline written to %s (%d kernels)" % (args.baseline, len(now)))
        return 0
    base = json.loads(args.baseline.read_text())["kernels"] if args.baseline.is_file() else {}
    if args.kernels:
        base = {k: v for k, v in base.items() if k in names}
    rows = compare(now, base, args.tolerance)
    print(table(rows))
    if args.markdown:
        bad = [r for r in rows if not r["ok"]]
        args.markdown.write_text("### atalla-sim cycles on the reference kernels\n\n%s\n\n%s\n"
                                 % (table(rows, markdown=True),
                                    "%d kernel(s) changed: update tests/perf_baseline.json "
                                    "(tools/bench.py --update) if that is intended." % len(bad)
                                    if bad else "No cycle count changed."))
    bad = [r["kernel"] for r in rows if not r["ok"]]
    if bad:
        print("\ncycles changed: %s. If intended, run `atalla-sim bench --update` and commit "
              "tests/perf_baseline.json with the change." % ", ".join(bad))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
