#!/usr/bin/env python3
"""Run programs on today's lane vector core and on the M x N core, side by side.

Each program is built once and the same image runs on both, so cycle counts
compare the vector cores and not two compiles (the compiler's register
allocation is not deterministic). For every run it reports cycles, whether the
result is right, and for the M x N core where its slots waited.

    tools/compare_vector_arch.py relu.c add.c softmax.c
    tools/compare_vector_arch.py relu.c --golden          # also bit-compare with the functional sim
    tools/compare_vector_arch.py softmax.c --wb-sweep 1,2,4,8
    tools/compare_vector_arch.py gemm prog.in             # functional-sim kernels and images too

docs/vector_core_v2_change_plan.md has the design; docs/vector_core_v2_worklog.md
the state of the model.
"""

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import c_kernels                                                  # noqa: E402
import run_kernel                                                 # noqa: E402
from atalla.atalla_platform import build_atalla_platform          # noqa: E402
from scheduler import golden                                      # noqa: E402
from scheduler.core import load_program_text                      # noqa: E402
from vector_core.v2.config import VectorCoreConfig                # noqa: E402


def load(name: str, tmp: Path):
    """(instr, data, init_sregs, c_path) for a .c program, a .in image or a
    functional-sim kernel name."""
    if name.endswith(".c"):
        c_path = run_kernel.find_c(name)
        program, data, sregs = c_kernels.build(c_path)
        return program.instr, data, sregs, c_path
    path = Path(name)
    if path.suffix != ".in":
        path = run_kernel.build_kernel(name, [], tmp)
    instr, data = load_program_text(path.read_text())
    return instr, data, None, None


def run_one(instr, data, sregs, c_path, cfg, gold, limit):
    plat = build_atalla_platform(instr, data, init_sregs=sregs, strict=False,
                                 vector_cfg=cfg, perf=True)
    cycles = plat.run_until_done(limit=limit)
    ok = "-"
    if c_path is not None:
        res, _msg = c_kernels.check(c_path, plat)
        ok = {True: "PASS", False: "FAIL", None: "n/a"}[res]
    diffs = None
    if gold is not None:
        diffs = len(golden.differences(plat, gold))
    return plat, cycles, ok, diffs


def v2_summary(vc) -> str:
    s = vc.stats
    stall = {k: sum(v) for k, v in s["stall"].items() if sum(v)}
    parts = ["fires %d" % sum(s["fires"]), "rounds %d" % s["rounds"]]
    if stall:
        parts.append("slot stalls " + ", ".join("%s %d" % kv for kv in sorted(stall.items())))
    xr = sum(s["compat_cross_pair_reads"])
    if xr:
        parts.append("cross-pair reads %d (+%d cycles)"
                     % (xr, sum(s["compat_cross_pair_cycles"])))
    denied = sum(vc.arbiter.stats["denied"].values())
    if denied:
        parts.append("wb denied %d" % denied)
    return "; ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("programs", nargs="+")
    ap.add_argument("--golden", action="store_true",
                    help="also compare every run's final state with the functional sim")
    ap.add_argument("--read-mode", choices=["compat", "strict"], default="compat")
    ap.add_argument("--wb-sweep", default="4",
                    help="comma-separated writebacks per cycle for the M x N core (Q11)")
    ap.add_argument("--no-m0-port", action="store_true",
                    help="unmasked ops don't use mask bank 0's read port")
    ap.add_argument("--limit", type=int, default=5_000_000)
    args = ap.parse_args()
    if args.golden and not golden.HAVE_GOLDEN:
        sys.exit("functional sim unavailable: %s" % golden.load_error)
    sweep = [int(x) for x in args.wb_sweep.split(",") if x]

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in args.programs:
            instr, data, sregs, c_path = load(name, Path(tmp))
            gold = (golden.run_golden_state(instr, data, reductions="lanes", init_sregs=sregs)
                    if args.golden else None)
            _, c1, ok1, d1 = run_one(instr, data, sregs, c_path, None, gold, args.limit)
            rows.append((name, "lanes", c1, ok1, d1, ""))
            for g in sweep:
                cfg = VectorCoreConfig(arch="fu_groups", read_mode=args.read_mode,
                                       wb_grants_per_cycle=g,
                                       m0_uses_port=not args.no_m0_port)
                plat, c2, ok2, d2 = run_one(instr, data, sregs, c_path, cfg, gold, args.limit)
                label = "fu_groups" + (" wb=%d" % g if len(sweep) > 1 else "")
                rows.append((name, label, c2, ok2, d2, v2_summary(plat.vc)))

    print("%-16s %-14s %9s %8s %7s  %s" % ("program", "vector core", "cycles", "result",
                                         "golden", "M x N detail"))
    base = {}
    for name, arch, cyc, ok, diffs, detail in rows:
        if arch == "lanes":
            base[name] = cyc
            delta = ""
        else:
            delta = " (%+.1f%%)" % (100.0 * (cyc - base[name]) / base[name]) if base[name] else ""
        g = "-" if diffs is None else ("same" if diffs == 0 else "%d diff" % diffs)
        print("%-16s %-14s %9s %8s %7s  %s" % (name, arch, "%d%s" % (cyc, delta), ok, g, detail))


if __name__ == "__main__":
    main()
