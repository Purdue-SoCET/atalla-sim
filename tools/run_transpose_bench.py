#!/usr/bin/env python3
"""Runtime of the transpose unit on one rows x 32 tile.

Two benches (src/atalla/transpose_bench.py):

  vrf   VRF -> transpose -> VRF
  spad  scratchpad -> VLSU -> VRF -> transpose -> VRF -> VLSU -> scratchpad

Examples:
  python tools/run_transpose_bench.py                     # both, 32-row tile
  python tools/run_transpose_bench.py --rows 8 16 32
  python tools/run_transpose_bench.py --bench spad --no-overlap
  python tools/run_transpose_bench.py --rows 1-32 --csv transpose.csv
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from atalla.transpose_bench import run_spad_bench, run_vrf_bench  # noqa: E402


def parse_rows(items):
    rows = []
    for item in items:
        if "-" in item:
            lo, hi = item.split("-", 1)
            rows.extend(range(int(lo), int(hi) + 1))
        else:
            rows.append(int(item))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", choices=("vrf", "spad", "both"), default="both")
    ap.add_argument("--rows", nargs="+", default=["32"],
                    help="tile heights, e.g. 8 16 32 or 1-32 (default 32)")
    ap.add_argument("--no-overlap", action="store_true",
                    help="spad: run load, transpose and store one after another")
    ap.add_argument("--load-window", type=int, default=4,
                    help="spad: max loads in flight (default 4; 0 = all in cycle 0)")
    ap.add_argument("--load-pad", type=int, default=0)
    ap.add_argument("--store-pad", type=int, default=0)
    ap.add_argument("--csv", help="write one row per run to this file")
    ap.add_argument("--json", help="write full results, with per-event cycles")
    ap.add_argument("-q", "--quiet", action="store_true", help="table only")
    args = ap.parse_args(argv)

    results = []
    for rows in parse_rows(args.rows):
        if args.bench in ("vrf", "both"):
            results.append(run_vrf_bench(rows))
        if args.bench in ("spad", "both"):
            results.append(run_spad_bench(
                rows, load_pad=args.load_pad, store_pad=args.store_pad,
                overlap=not args.no_overlap,
                load_window=args.load_window or None))

    if not args.quiet:
        for r in results:
            print(r.summary())
        print()
    print("%-5s %5s %8s" % ("bench", "rows", "cycles"))
    for r in results:
        print("%-5s %5d %8d" % (r.bench, r.rows, r.cycles))

    if args.csv:
        phases = sorted({p for r in results for p in r.phases})
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bench", "rows", "cycles"]
                       + ["%s_%s" % (p, e) for p in phases for e in ("start", "end")])
            for r in results:
                w.writerow([r.bench, r.rows, r.cycles]
                           + [x for p in phases for x in r.phases.get(p, ("", ""))])
    if args.json:
        with open(args.json, "w") as f:
            json.dump([r.as_dict() for r in results], f, indent=1)


if __name__ == "__main__":
    main()
