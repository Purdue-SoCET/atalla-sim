#!/usr/bin/env python3
"""Run one program over a grid of machine configurations: `atalla-sim sweep`.

    atalla-sim sweep gemm --param lanes=4,8,16,32
    atalla-sim sweep relu.c --param dram_latency=6,20,50 --param array=meissa,tpu -j 4
    atalla-sim sweep prog.in --config atalla-0x01-ddr4 --param ramulator.nCL=16,22 \\
        -o sweep.csv --plot sweep.png --perf

A parameter is any key of the machine configuration (configs/atalla-0x01.toml):
a dotted path (`dcache.size_bytes`, `memory.ramulator.nCL`), or a key name
that appears once (`lane_count`, `dram_latency`), or one of the short names
lanes, array, dram, latency. Values are comma-separated. Every combination
runs, in parallel with -j. The program is built (or compiled) once, so every
point runs the same image.

Each row has the parameters, the cycles, and the packets issued; --perf adds
the performance monitor's cycle buckets (it slows each run by about 7%).
-o writes CSV; --plot draws cycles against the first parameter, one line per
value of the second (needs matplotlib).
"""

import sys
if sys.version_info < (3, 9):
    sys.exit("atalla-sim needs Python 3.9 or newer; %s is %s"
             % (sys.executable, sys.version.split()[0]))

import argparse
import csv
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "src"), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from atalla import config                                          # noqa: E402
from atalla.atalla_platform import build_atalla_platform           # noqa: E402

ALIASES = {"lanes": "core.lane_count", "array": "core.systolic_array",
           "dram": "memory.dram", "latency": "memory.dram_latency"}


def _leaves(d: Dict[str, Any], prefix: Tuple[str, ...] = ()) -> List[Tuple[str, ...]]:
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out += _leaves(v, prefix + (k,))
        else:
            out.append(prefix + (k,))
    return out


def resolve_key(cfg: Dict[str, Any], key: str) -> Tuple[str, ...]:
    """A parameter name -> its path in the configuration."""
    key = ALIASES.get(key, key)
    if "." in key:
        path = tuple(key.split("."))
        if path[0] not in cfg:
            # memory.ramulator.nCL may be given as ramulator.nCL
            for top, sub in cfg.items():
                if isinstance(sub, dict) and path[0] in sub:
                    path = (top,) + path
                    break
        node = cfg
        for part in path[:-1]:
            if not isinstance(node.get(part), dict):
                raise ValueError("no configuration table %r" % ".".join(path[:-1]))
            node = node[part]
        open_table = path[:-1] in (("core", "ex_latency"), ("memory", "ramulator"))
        if path[-1] not in node and not open_table:
            raise ValueError("no configuration key %r" % key)
        return path
    hits = [p for p in _leaves(cfg) if p[-1] == key]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ValueError("no configuration key %r (see configs/atalla-0x01.toml)" % key)
    raise ValueError("%r is ambiguous: %s" % (key, ", ".join(".".join(h) for h in hits)))


def parse_value(text: str):
    t = text.strip()
    if t.lower() in ("true", "false"):
        return t.lower() == "true"
    for cast in (lambda s: int(s, 0), float):
        try:
            return cast(t)
        except ValueError:
            pass
    return t


def nest(path: Tuple[str, ...], value) -> Dict[str, Any]:
    out: Dict[str, Any] = value
    for part in reversed(path):
        out = {part: out}
    return out


def _deep(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep(out.get(k, {}), v) if isinstance(v, dict) else v
    return out


def run_point(job) -> Dict[str, Any]:
    instr, data, regs, cfg, perf, limit, label = job
    t = time.time()
    kw = config.platform_kwargs(cfg)
    plat = build_atalla_platform(instr, data, init_sregs=regs, perf=perf, strict=False, **kw)
    row: Dict[str, Any] = dict(label)
    try:
        row["cycles"] = plat.run_until_done(limit=limit)
        row["error"] = ""
    except RuntimeError as exc:
        row["cycles"] = ""
        row["error"] = str(exc).splitlines()[0]
    row["packets"] = plat.core.issued_count
    if perf:
        for bucket, n in sorted(plat.perf.buckets.items()):
            row["cycles:" + bucket] = n
    row["seconds"] = round(time.time() - t, 2)
    return row


def plot(rows: List[Dict[str, Any]], names: List[str], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    x = names[0]
    groups = (sorted({r.get(names[1]) for r in rows}, key=lambda v: (isinstance(v, str), v))
              if len(names) > 1 else [None])
    for g in groups:
        pts = [(r[x], r["cycles"]) for r in rows
               if r["cycles"] != "" and (g is None or r.get(names[1]) == g)]
        pts.sort(key=lambda p: (str(type(p[0])), p[0]))
        ax.plot([str(p[0]) if isinstance(p[0], str) else p[0] for p in pts],
                [p[1] for p in pts], marker="o",
                label=None if g is None else "%s = %s" % (names[1], g))
    ax.set_xlabel(x)
    ax.set_ylabel("cycles")
    ax.grid(True, alpha=0.3)
    if groups != [None]:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main(argv=None) -> int:
    import run_kernel
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: List[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(prog="atalla-sim sweep",
                                 description=__doc__.split("\n\n")[0],
                                 epilog="Parameters: " + __doc__.split("\n\n")[2])
    ap.add_argument("kernel", help="a kernel name, a .c program, a .in or a .elf")
    ap.add_argument("--param", "-p", metavar="KEY=V1,V2,...", action="append", default=[],
                    required=True, help="a configuration key and its values; repeatable")
    ap.add_argument("--config", metavar="NAME|PATH", help="the base machine (default atalla-0x01)")
    ap.add_argument("--elf", action="store_true", help="for a .c program: run its linked ELF")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="points run at once (default 1)")
    ap.add_argument("--limit", type=int, default=5_000_000, help="cycle limit per point")
    ap.add_argument("--perf", action="store_true", help="add the monitor's cycle buckets")
    ap.add_argument("-o", "--output", metavar="CSV", help="write the table as CSV")
    ap.add_argument("--plot", metavar="PNG", help="draw cycles against the first parameter")
    args = ap.parse_args(argv)

    try:
        base = config.load(args.config)
        axes = []
        for spec in args.param:
            key, eq, values = spec.partition("=")
            if not eq or not values:
                raise ValueError("--param takes KEY=V1,V2,...; not %r" % spec)
            axes.append((key, resolve_key(base, key), [parse_value(v) for v in values.split(",")]))
    except (ValueError, config.ConfigError) as exc:
        sys.exit(str(exc))
    names = [k for k, _, _ in axes]

    loaded = run_kernel.load_program(args.kernel, extra, args.elf)
    jobs = []
    for combo in itertools.product(*[vals for _, _, vals in axes]):
        change: Dict[str, Any] = {}
        for (_, path, _), value in zip(axes, combo):
            change = _deep(change, nest(path, value))
        try:
            cfg = config.override(base, change)
        except config.ConfigError as exc:
            sys.exit(str(exc))
        jobs.append((loaded.instr, loaded.data, loaded.init_sregs, cfg, args.perf, args.limit,
                     tuple(zip(names, combo))))
    print("%s: %d point%s on %s" % (args.kernel, len(jobs), "s" * (len(jobs) != 1),
                                    config.summary(base)))
    if args.jobs > 1 and len(jobs) > 1:
        import multiprocessing
        ctx = multiprocessing.get_context("fork" if os.name == "posix" else "spawn")
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as pool:
            rows = list(pool.map(run_point, jobs))
    else:
        rows = [run_point(j) for j in jobs]

    cols = names + ["cycles", "packets"] + sorted(
        {k for r in rows for k in r if k.startswith("cycles:")}) + ["seconds", "error"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    shown = [c for c in cols if c != "error" or any(r["error"] for r in rows)]
    print("  ".join("%*s" % (widths[c], c) for c in shown))
    for r in rows:
        print("  ".join("%*s" % (widths[c], r.get(c, "")) for c in shown))
    if args.output:
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
        print("table written to %s" % args.output)
    if args.plot:
        try:
            plot(rows, names, args.plot)
            print("plot written to %s" % args.plot)
        except ImportError:
            print("--plot needs matplotlib (pip install matplotlib)")
    return 1 if any(r["error"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
