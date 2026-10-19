#!/usr/bin/env python3
"""Run a kernel on the scheduler-driven platform.

The instruction stream goes to the VLIW scheduler model (fetch, decode,
issue, scalar execute), which drives the vector core's units, the systolic
array, the four scratchpad pads with their DMA backends, and DRAM
(src/atalla/atalla_platform.py).

    tools/run_kernel.py gemm                 # build kernels/build_gemm.py, run it
    tools/run_kernel.py path/to/prog.in      # an already assembled program
    tools/run_kernel.py softmax --golden     # also run the functional sim and compare
    tools/run_kernel.py gemm -- --rows 16    # arguments after -- go to the build script
    tools/run_kernel.py gemm --perf-json gemm.json --timeline   # the monitor's report

A kernel name is built with the functional sim's own build script,
third_party/atalla-functional-sim/kernels/build_<name>.py.

A C program (a .c path, or a file name in the compiler's
test_validation_kernels/ or test_validation/) goes straight from the compiler
to atalla-sim: atalla_cc packetizes it, src/scheduler/assembler.py assembles
it, and its inputs and reference result come from the compiler's validation
harness (tools/c_kernels.py). It runs with the C loader's stack registers set.

    tools/run_kernel.py add.c                  # run, check against numpy
    tools/run_kernel.py relu.c --golden        # also compare with the functional sim
    tools/run_kernel.py add.c --save add.in    # keep the assembled image
    tools/run_kernel.py gemm --activity gemm.html   # which blocks work when
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scheduler import golden                                     # noqa: E402
from scheduler.core import load_program_text                      # noqa: E402
from atalla.atalla_platform import build_atalla_platform           # noqa: E402
from scheduler.assembler import to_in_text                        # noqa: E402
from memory.dram import DRAM_BURST_BYTES                          # noqa: E402
import c_kernels                                                  # noqa: E402

SIM = golden.SUBMODULE_PATH


def _sim_env():
    """The environment the functional sim's build scripts need."""
    paths = [str(SIM), str(SIM / "kernels"), str(SIM / "experiments" / "gemm")]
    return dict(os.environ, PYTHONPATH=os.pathsep.join(paths))


def find_c(name: str) -> Path:
    """A .c path as given, or by file name among the compiler's programs."""
    path = Path(name)
    if path.is_file():
        return path.resolve()
    for folder in (c_kernels.KERNELS, c_kernels.UNIT_TESTS):
        if (folder / path.name).is_file():
            return folder / path.name
    sys.exit("no C program %s (looked in %s and %s)"
             % (name, c_kernels.KERNELS, c_kernels.UNIT_TESTS))


def build_kernel(name: str, extra, out_dir: Path) -> Path:
    script = SIM / "kernels" / ("build_%s.py" % name)
    if not script.is_file():
        sys.exit("no build script %s" % script)
    out = out_dir / ("%s.in" % name)
    p = subprocess.run([sys.executable, str(script), "-o", str(out), *extra],
                       cwd=str(SIM), env=_sim_env(),
                       capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit("build failed:\n" + (p.stderr or p.stdout)[-3000:])
    return out


def compare(plat, instr, data, reductions: str = "lanes", init_sregs=None) -> int:
    """Run the functional sim on the same program and print where the
    platform's final state differs (golden.differences). Returns how many.
    reductions="lanes": the functional sim reduces in the lane model's
    order; "functional": its own sequential fp32 sum."""
    g = golden.run_golden_state(instr, data, reductions=reductions, init_sregs=init_sregs)
    diffs = golden.differences(plat, g)
    for line in diffs[:20]:
        print("  " + line)
    print("  %d differences" % len(diffs) if diffs else "  identical: registers and DRAM")
    return len(diffs)


def main() -> None:
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("kernel", help="a kernel name (kernels/build_<name>.py), a .c program "
                                   "or a .in file")
    ap.add_argument("--golden", action="store_true",
                    help="also run the functional sim and compare the final state")
    ap.add_argument("--limit", type=int, default=5_000_000, help="cycle limit")
    ap.add_argument("--dram-latency", type=int, default=6,
                    help="cycles from a DRAM request launching to its data returning")
    ap.add_argument("--lanes", type=int, default=16, help="vector lanes (must divide 32)")
    ap.add_argument("--array", choices=["meissa", "tpu"], default="meissa",
                    help="the systolic array: MEISSA (the RTL's) or the TPU-style array")
    ap.add_argument("--burst-bytes", type=int, default=DRAM_BURST_BYTES,
                    help="bytes per DRAM request, for every DRAM master "
                         "(default: the RTL's 64-bit interface)")
    ap.add_argument("--warm-icache", action="store_true",
                    help="start with every packet's line in the icache")
    ap.add_argument("--functional-reductions", action="store_true",
                    help="with --golden: compare against the functional sim's own "
                         "fp32 sums instead of the lane model's reduction order")
    ap.add_argument("--save", metavar="PATH",
                    help="write the program image (.in) that ran, e.g. for the functional sim")
    ap.add_argument("--activity", metavar="PATH",
                    help="record which blocks are busy each cycle: a .html path draws the "
                         "activity view (tools/activity_view.py), anything else saves the trace")
    ap.add_argument("--perf-json", metavar="PATH",
                    help="write the performance monitor's report as JSON")
    ap.add_argument("--timeline", action="store_true",
                    help="with --perf-json: include the per-cycle timeline")
    args = ap.parse_args(argv)

    init_sregs = None
    c_path = None
    if args.kernel.endswith(".c"):
        if extra:
            sys.exit("a C program takes no build arguments (its inputs come from the "
                     "compiler's harness)")
        c_path = find_c(args.kernel)
        try:
            program, data, init_sregs = c_kernels.build(c_path)
        except (c_kernels.CompileError, ValueError) as exc:
            sys.exit(str(exc))
        instr = program.instr
    else:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(args.kernel)
            if path.suffix != ".in":
                path = build_kernel(args.kernel, extra, Path(tmp))
            instr, data = load_program_text(path.read_text())
    if args.save:
        Path(args.save).write_text(to_in_text(instr, data))
        print("program image written to %s" % args.save)

    plat = build_atalla_platform(instr, data, dram_latency=args.dram_latency,
                                 lane_count=args.lanes, systolic_array=args.array,
                                 burst_bytes=args.burst_bytes,
                                 activity=bool(args.activity), warm_icache=args.warm_icache,
                                 perf_timeline=args.timeline, init_sregs=init_sregs,
                                 strict=False)
    t = time.time()
    cycles = plat.run_until_done(limit=args.limit)
    c = plat.core
    print("%s: %d cycles (%.1f s)" % (args.kernel, cycles, time.time() - t))
    print()
    print(plat.perf.summary())
    print()
    if args.perf_json:
        plat.perf.to_json(args.perf_json, timeline=args.timeline)
        print("performance report written to %s" % args.perf_json)
    if args.activity:
        meta = dict(program=Path(args.kernel).name, array=args.array, lanes=args.lanes,
                    burst_bytes=args.burst_bytes, dram_latency=args.dram_latency)
        if args.activity.endswith(".html"):
            import activity_view
            Path(args.activity).write_text(activity_view.render(plat.activity.trace(meta)))
            print("activity view written to %s" % args.activity)
        else:
            plat.activity.to_json(args.activity, meta)
            print("activity trace written to %s (tools/activity_view.py draws it)" % args.activity)
    if c.decode2.violations:
        print("  packets the RTL would mangle: %d (first at pc %#x: %s)"
              % (len(c.decode2.violations), c.decode2.violations[0][0],
                 "; ".join(c.decode2.violations[0][1])))
    if c_path is not None:
        ok, message = c_kernels.check(c_path, plat)
        print("reference result: %s -- %s"
              % ({True: "PASS", False: "FAIL", None: "not checked"}[ok], message))
        bug = c_kernels.known_bug(c_path)
        if bug and not ok:
            print("  a known compiler bug: %s" % bug)
    if args.golden:
        if not golden.HAVE_GOLDEN:
            sys.exit("functional sim unavailable: %s" % golden.load_error)
        red = "functional" if args.functional_reductions else "lanes"
        print("against the functional sim (%s reduction order):" % red)
        compare(plat, instr, data, red, init_sregs)


if __name__ == "__main__":
    main()
