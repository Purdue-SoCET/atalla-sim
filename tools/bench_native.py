#!/usr/bin/env python3
"""Time the systolic-array datapath with and without the native SIMD kernels.

    python tools/bench_native.py [--size 32] [--ticks 400]

Runs the same steady-state workload through the native path and the numpy
fallback and prints microseconds per simulated cycle for each. Build the
kernels first with `make native`; without them only the fallback is timed.
"""

import argparse
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def measure(size, group_size, ticks):
    from base.clock_domain import ClockDomain
    from base.core import Core
    from base.eventq import EventQueue
    from base.sim import Sim
    from systolic_array.systolic_array_tpu import SystolicArrayTPU

    eq = EventQueue()
    clk = ClockDomain(eq, period=1.0)
    core = Core(eq)
    core.add_clock_domain(clk)
    sim = Sim()
    sim.init(eq, core)

    sa = SystolicArrayTPU(size, boundary_buffer_depth=64, dtype="fp16",
                          group_size=group_size)
    clk.add_clocked(sa)

    rng = random.Random(99)
    sa.load_weights([[rng.uniform(-4, 4) for _ in range(size)] for _ in range(size)])
    sa.set_control(weight_en=False, mac_shift=True, start=True, stall=False)

    rows = [[rng.uniform(-4, 4) for _ in range(size)] for _ in range(64)]
    psums = [[rng.uniform(-4, 4) for _ in range(size)] for _ in range(64)]

    warmup = max(10, ticks // 20)
    for i in range(warmup):
        sa.enqueue(rows[i % 64])
        sa.enqueue_psums(psums[i % 64])
        sa.tick(float(i))

    t0 = time.perf_counter()
    for i in range(warmup, warmup + ticks):
        sa.enqueue(rows[i % 64])
        sa.enqueue_psums(psums[i % 64])
        sa.tick(float(i))
    return (time.perf_counter() - t0) / ticks * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--ticks", type=int, default=400)
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        # Timed in a subprocess so ATALLA_NO_NATIVE is applied at import time.
        print("%.3f" % measure(args.size, args.group_size, args.ticks))
        return

    from native import kernels

    def child(no_native):
        env = dict(os.environ)
        env["ATALLA_NO_NATIVE"] = "1" if no_native else "0"
        out = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--child",
             "--size", str(args.size), "--group-size", str(args.group_size),
             "--ticks", str(args.ticks)],
            capture_output=True, text=True, env=env, check=True)
        return float(out.stdout.strip().splitlines()[-1])

    print(f"systolic array {args.size}x{args.size}, group_size={args.group_size}, "
          f"fp16, {args.ticks} ticks")
    fallback = child(True)
    print(f"  numpy fallback   {fallback:9.1f} us / simulated cycle")
    if kernels.HAVE_NATIVE:
        nat = child(False)
        print(f"  native ({kernels.NATIVE_ISA:<9}) {nat:9.1f} us / simulated cycle"
              f"   [{fallback / nat:.1f}x faster]")
    else:
        print(f"  native            unavailable ({kernels.load_error});"
              f" build with `make native`")


if __name__ == "__main__":
    main()
