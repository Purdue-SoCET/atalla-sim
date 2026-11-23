#!/usr/bin/env python3
"""A CISC vector loop against a baseline kernel, over tile sizes, the loop's
variants and the baseline's core: cycles, speedup, accuracy, and the
buffer-area / load trade.

    tools/sweep_vloop.py                                   # relu, rows 1 2 4 8 16 32
    tools/sweep_vloop.py --kernel gelu                     # gelu.vl vs gelu.c
    tools/sweep_vloop.py --kernel gelu --baseline asm      # vs build_gelu.py instead
    tools/sweep_vloop.py --kernel gelu --variant all       # keep / reload / reload_all
    tools/sweep_vloop.py --kernel gelu --baseline-arch lanes   # baseline on today's core only
    tools/sweep_vloop.py --kernel gelu --rows 4 32 --golden --csv gelu.csv
    tools/sweep_vloop.py --dram ramulator                  # DRAM timed by Ramulator 2

Programs, for --kernel K (both get the same data for the same --rows /
--seed / --scale; the tile is as wide as the baseline's):
    vloop          kernels/build_vloop.py --loop K.vl   always on the M x N core
    --baseline asm kernels/build_K.py (functional sim)  ordinary instructions
    --baseline c   a C kernel, tools/c_tile_kernels.C_KERNELS[K] (gelu.c),
                   compiled with the newer compiler in aihw-ppci-compiler/
The default is c where there is one, else asm. --baseline-arch runs the
baseline on the lane core ("lanes", today's Atalla), the M x N core
("fu_groups"), or both (default), so the M x N core's gain and the loop's
gain show apart. The vloop runs only on the M x N core: gelu.vl's EXP and
RECIP groups exist only there. Any loop in vector_core/vloop.LOOPS with a
baseline works; the variants (vloop.variants_of) are the sequencer's
setting, so the vloop program is the same for all of them. The icache is
warm, so fetch doesn't count. Every run is strict: a packet the RTL can't
execute raises.

Columns, per rows and variant:
  base_<arch>      the baseline's cycles on that core; speedup_<arch> = it / vloop
  vloop            the loop kernel's cycles
  err_base/err_vloop  max |output - the true function| (relu; tanh GELU)
  ii               the loop's II, "6(4)" when the pad's reads raised it from 4
  loop_cycles, loop_stalls, rows/run_cycle against the 1/II ceiling,
  loads/row, buffer_bits (steady-state buffer size), sdma_cycles
  diffs_*          with --golden, differences from the functional sim. Not for
                   a C baseline: the functional sim takes branch offsets x4,
                   atalla-sim x5, so compiled C diverges there (known).
"""

import argparse
import csv
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import c_tile_kernels as ct                                 # noqa: E402
from atalla.atalla_platform import build_atalla_platform, c_entry_sregs  # noqa: E402
from memory.ramulator import RamulatorDRAM                  # noqa: E402
from scheduler import golden                                # noqa: E402
from scheduler.core import load_program_text                # noqa: E402
from run_kernel import SIM, build_kernel                    # noqa: E402
from vector_core.v2.config import VectorCoreConfig          # noqa: E402
from vector_core.vloop import LOOPS, variants_of            # noqa: E402

ARCHES = {"lanes": "lanes", "fu_groups": "mxn"}
#: The function each kernel computes, for the accuracy columns.
TRUE = {"relu": lambda x: max(x, 0.0), "gelu": ct.tanh_gelu}


def baselines(kernel):
    """The baselines a kernel has, default first."""
    out = []
    if kernel in ct.C_KERNELS and ct.C_KERNELS[kernel].path.is_file():
        out.append("c")
    if (SIM / "kernels" / ("build_%s.py" % kernel)).is_file():
        out.append("asm")
    return out


#: kernel name ("gelu") -> its loop's funct, for loops with a baseline.
KERNELS = {name.split(".")[0]: f for f, (name, _, _) in LOOPS.items()
           if baselines(name.split(".")[0])}


class Program:
    """A built program: image, data, initial registers, tile width."""
    def __init__(self, instr, data, regs, width):
        self.instr, self.data, self.regs, self.width = instr, data, regs, width


def build(kernel, kind, rows, args, tmp):
    """kind "c", "asm" or "vloop" -> Program. `args.width` must already be
    the baseline's width."""
    if kind == "c":
        k = ct.C_KERNELS[kernel]
        data, width = k.image(k.path, rows, args.seed, args.scale)
        return Program(ct.program(kernel).instr, data, c_entry_sregs(data), width)
    name = "vloop" if kind == "vloop" else kernel
    extra = ["--rows", str(rows), "--width", str(args.width), "--seed", str(args.seed)]
    if takes_scale(name):
        extra += ["--scale", str(args.scale)]
    elif args.scale != 1.0:
        raise SystemExit("build_%s.py has no --scale (its data is unscaled)" % name)
    if kind == "vloop":
        extra += ["--loop", LOOPS[KERNELS[kernel]][0]]
    instr, data = load_program_text(build_kernel(name, extra, Path(tmp)).read_text())
    return Program(instr, data, None, args.width)


def takes_scale(name):
    """Whether kernels/build_<name>.py takes --scale (the tile_kernel ones do;
    build_relu.py, older, builds the same data at scale 1 without it)."""
    text = (SIM / "kernels" / ("build_%s.py" % name)).read_text()
    return "tile_args" in text or "--scale" in text


def run(prog, arch, args, variants=None, check_golden=False):
    """(cycles, perf report, golden diffs or None, platform)."""
    cfg = VectorCoreConfig(arch=arch, read_mode="compat") if arch == "fu_groups" else None
    plat = build_atalla_platform(
        prog.instr, prog.data, dram_latency=args.dram_latency, warm_icache=True,
        dram_channel=RamulatorDRAM("rtl") if args.dram == "ramulator" else None,
        vector_cfg=cfg, init_sregs=prog.regs, vloop_variants=variants)
    cycles = plat.run_until_done(limit=args.limit)
    diffs = None
    if check_golden:
        diffs = len(golden.differences(plat, golden.run_golden_state(
            prog.instr, prog.data, reductions="lanes", init_sregs=prog.regs)))
    return cycles, plat.perf.report(), diffs, plat


def max_error(plat, kernel, rows, args, width):
    """max |output - true function| over the tile, of the inputs as stored
    in DRAM (the kernels leave them there)."""
    x = ct.read_tile(plat, ct.INPUT_BASE, rows, width)
    out = ct.read_tile(plat, ct.OUTPUT_BASE, rows, width)
    f = TRUE[kernel]
    errs = [abs(out[r][c] - f(x[r][c])) for r in range(rows) for c in range(width)]
    return max(e if not math.isnan(e) else math.inf for e in errs)


def loop_columns(rep):
    loop = rep.get("vloop") or {}
    table = next(iter(loop.get("tables", {}).values()), {})
    ii = table.get("ii")
    if ii is not None and table.get("ii_wanted") not in (None, ii):
        ii = "%d(%d)" % (ii, table["ii_wanted"])
    sdma = sum(v["cycles"] for k, v in rep["unit_busy"].items() if k.startswith("sdma"))
    return {"ii": ii, "loop_cycles": loop.get("cycles", {}).get("total", 0),
            "loop_stalls": loop.get("stall_cycles", 0),
            "rows/run_cycle": loop.get("rows_per_run_cycle", 0.0),
            "ceiling": loop.get("rows_per_cycle_ceiling"),
            "loads/row": loop.get("loads_per_row", 0.0),
            "buffer_bits": loop.get("buffer_bits_total", 0), "sdma_cycles": sdma}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kernel", default="relu", choices=sorted(KERNELS),
                    help="the function: its loop K.vl against a baseline (default relu)")
    ap.add_argument("--baseline", choices=["c", "asm"], default=None,
                    help="the baseline: a C kernel or the functional sim's build_K.py "
                         "(default c where there is one)")
    ap.add_argument("--baseline-arch", nargs="+", choices=list(ARCHES),
                    default=list(ARCHES), help="the core(s) the baseline runs on (default both)")
    ap.add_argument("--variant", nargs="+", default=["keep"],
                    help="the loop's variants to run, or 'all' (default keep)")
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32],
                    help="tile rows to try (1..32)")
    ap.add_argument("--width", type=int, default=32,
                    help="tile columns for an asm baseline (a C kernel sets its own)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=None,
                    help="input standard deviation (default: the kernel's)")
    ap.add_argument("--dram", choices=["fixed", "ramulator"], default="fixed")
    ap.add_argument("--dram-latency", type=int, default=6)
    ap.add_argument("--limit", type=int, default=5_000_000, help="cycle limit per run")
    ap.add_argument("--golden", action="store_true",
                    help="compare each run with the functional sim (not a C baseline)")
    ap.add_argument("--csv", metavar="PATH", help="also write the table as CSV")
    args = ap.parse_args()

    kernel, funct = args.kernel, KERNELS[args.kernel]
    kind = args.baseline or baselines(kernel)[0]
    if kind not in baselines(kernel):
        ap.error("%s has no %s baseline (has %s)" % (kernel, kind, ", ".join(baselines(kernel))))
    if args.scale is None:
        args.scale = ct.DEFAULT_SCALE.get(kernel, 1.0)
    if kind == "c":
        k = ct.C_KERNELS[kernel]
        try:
            args.width = ct.tile_width(k.path)
            ct.program(kernel)                      # compile now, so errors come first
        except ValueError as e:
            ap.error(str(e))
        except Exception as e:                      # c_kernels.CompileError
            last = str(e).strip().splitlines()[-1] if str(e).strip() else repr(e)
            ap.error("compiling %s failed: %s" % (k.path.name, last))
        print("baseline %s (%d columns, compiled with %s)" % (k.path.name, args.width,
                                                             ct.C_COMPILER.name))
        if kernel == "gelu":
            print("gelu.c coefficients c6 c4 c2 c0 = %s" % ", ".join("%g" % c for c in ct.GELU_POLY))
    known = list(variants_of(funct))
    variants = known if args.variant == ["all"] else args.variant
    bad = [v for v in variants if v not in known]
    if bad:
        ap.error("%s has no variant %s (has %s)"
                 % (LOOPS[funct][0], ", ".join(bad), ", ".join(known)))

    gold_base = args.golden and kind != "c"
    head = (["rows", "variant"] + ["base_" + ARCHES[a] for a in args.baseline_arch] + ["vloop"]
            + ["speedup_" + ARCHES[a] for a in args.baseline_arch] + ["err_base", "err_vloop"]
            + ["ii", "loop_cycles", "loop_stalls", "rows/run_cycle", "ceiling", "loads/row",
               "buffer_bits", "sdma_cycles"]
            + (["diffs_base", "diffs_vloop"] if args.golden else []))
    table = []
    with tempfile.TemporaryDirectory() as tmp:
        for rows in args.rows:
            base_prog = build(kernel, kind, rows, args, tmp)
            vl_prog = build(kernel, "vloop", rows, args, tmp)
            base, err_base, d0 = {}, None, None
            for arch in args.baseline_arch:
                cyc, _, d, plat = run(base_prog, arch, args, check_golden=gold_base)
                base[arch] = cyc
                err_base = max_error(plat, kernel, rows, args, base_prog.width)
                d0 = d if d0 is None else d0
            for variant in variants:
                vl, rep, d1, plat = run(vl_prog, "fu_groups", args, {funct: variant},
                                        check_golden=args.golden)
                row = {"rows": rows, "variant": variant, "vloop": vl,
                       "err_base": err_base,
                       "err_vloop": max_error(plat, kernel, rows, args, vl_prog.width),
                       **loop_columns(rep), "diffs_base": d0, "diffs_vloop": d1}
                for arch, cyc in base.items():
                    row["base_" + ARCHES[arch]] = cyc
                    row["speedup_" + ARCHES[arch]] = cyc / vl if vl else float("nan")
                table.append([row.get(h) for h in head])
                print(_fmt(head, table[-1], header=len(table) == 1), flush=True)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(head)
            w.writerows(table)
        print("wrote %s" % args.csv)


def _fmt(head, row, header):
    def cell(v):
        if isinstance(v, float):
            return "%.4f" % v if abs(v) < 1 else "%.3f" % v
        return "-" if v is None else str(v)
    widths = [max(len(h), 8) for h in head]
    widths[1] = max(widths[1], 10)
    line = "  ".join(cell(v).rjust(w) for v, w in zip(row, widths))
    if not header:
        return line
    return "  ".join(h.rjust(w) for h, w in zip(head, widths)) + "\n" + line


if __name__ == "__main__":
    main()
