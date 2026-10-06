#!/usr/bin/env python3
"""Run a kernel on the scheduler-driven platform.

The instruction stream goes to the VLIW scheduler model (fetch, decode,
issue, scalar execute), which drives the vector core's units, the systolic
array, the four scratchpad pads with their DMA backends, and DRAM
(src/scheduler/platform.py).

    tools/run_kernel.py gemm                 # build kernels/build_gemm.py, run it
    tools/run_kernel.py path/to/prog.in      # an already assembled program
    tools/run_kernel.py softmax --golden     # also run the functional sim and compare
    tools/run_kernel.py gemm -- --rows 16    # arguments after -- go to the build script

A kernel name is built with the functional sim's own build script,
third_party/atalla-functional-sim/kernels/build_<name>.py.
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
from scheduler.platform import build_scheduler_platform           # noqa: E402
from scheduler.semantics import u32                               # noqa: E402
from scheduler.vector import bf16_round                           # noqa: E402

SIM = golden.SUBMODULE_PATH


def build_kernel(name: str, extra, out_dir: Path) -> Path:
    script = SIM / "kernels" / ("build_%s.py" % name)
    if not script.is_file():
        sys.exit("no build script %s" % script)
    out = out_dir / ("%s.in" % name)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(SIM), str(SIM / "kernels"), str(SIM / "experiments" / "gemm")]))
    p = subprocess.run([sys.executable, str(script), "-o", str(out), *extra],
                       cwd=str(SIM), env=env, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit("build failed:\n" + (p.stderr or p.stdout)[-3000:])
    return out


def compare(plat, instr, data) -> int:
    """Differences from the functional sim: scalar and vector registers, and
    DRAM as the DMA reads it. Returns how many."""
    from src.misc.memory import Memory
    g = golden.run_golden_state(instr, data)
    c, bad = plat.core, 0
    for r in range(1, 256):
        if c.scalar_reg(r) != u32(g["sregs"].get(r, 0)):
            bad += 1
            print("  x%-3d model %#010x  functional sim %#010x"
                  % (r, c.scalar_reg(r), u32(g["sregs"].get(r, 0))))
    for v in range(256):
        if plat.vc.read_vreg(v) != [bf16_round(x) for x in g["vregs"][v]]:
            bad += 1
            print("  v%-3d differs" % v)
    gm = Memory()
    gm.data_mem = dict(g["mem"])
    halves = sorted({(a & ~3) + k for a in g["mem"] for k in (0, 2)})
    dram = [b for b in halves
            if int.from_bytes(plat.dram.read(b, 2), "little") != gm.read_bf16_le(b)]
    bad += len(dram)
    print("  DRAM: %d of %d BF16 halfwords differ" % (len(dram), len(halves)))
    return bad


def main() -> None:
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("kernel", help="a kernel name (kernels/build_<name>.py) or a .in file")
    ap.add_argument("--golden", action="store_true",
                    help="also run the functional sim and compare the final state")
    ap.add_argument("--limit", type=int, default=5_000_000, help="cycle limit")
    ap.add_argument("--dram-latency", type=int, default=6)
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--cold-icache", action="store_true")
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(args.kernel)
        if not path.suffix == ".in":
            path = build_kernel(args.kernel, extra, Path(tmp))
        instr, data = load_program_text(path.read_text())

    plat = build_scheduler_platform(instr, data, dram_latency=args.dram_latency,
                                    lane_count=args.lanes,
                                    warm_icache=not args.cold_icache, strict=False)
    t = time.time()
    cycles = plat.run_until_done(limit=args.limit)
    c = plat.core
    print("%s: %d cycles (%.1f s)" % (args.kernel, cycles, time.time() - t))
    print("  packets issued %d, fetched %d, flushes %d" %
          (len(c.issued), c.packets_fetched, c.flushes))
    if c.decode2.stall_reasons:
        print("  decode 2 stalls: %s" % ", ".join(
            "%s %d" % kv for kv in sorted(c.decode2.stall_reasons.items())))
    print("  vector side: %s" % ", ".join("%s %d" % kv for kv in c.vector.stats.items() if kv[1]))
    print("  load/store unit: %s" % ", ".join("%s %d" % kv for kv in c.lsu.stats.items() if kv[1]))
    print("  data cache: %s" % ", ".join("%s %d" % kv for kv in c.dcache.stats.items() if kv[1]))
    if c.decode2.violations:
        print("  packets the RTL would mangle: %d (first at pc %#x: %s)"
              % (len(c.decode2.violations), c.decode2.violations[0][0],
                 "; ".join(c.decode2.violations[0][1])))
    if args.golden:
        if not golden.HAVE_GOLDEN:
            sys.exit("functional sim unavailable: %s" % golden.load_error)
        print("against the functional sim:")
        compare(plat, instr, data)


if __name__ == "__main__":
    main()
