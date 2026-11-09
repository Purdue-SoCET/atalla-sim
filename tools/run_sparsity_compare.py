#!/usr/bin/env python3
"""Race the 2:4 sparse array against the regular one on the SAME pruned
workload, and print what the array-level result does and does not settle.

    python tools/run_sparsity_compare.py
    python tools/run_sparsity_compare.py --size 32 --csv out.csv

Both arrays run the same 2:4-pruned weights. The regular one ignores the
zeros and takes ceil(K/size) passes; the sparse one packs two of every four
and takes ceil(K/2size). Cycles include each pass's weight load.

WHAT THIS MEASURES, AND WHAT IT DOES NOT

Driving the arrays directly, every pass costs the same and the weight load is
`size` pushes either way, so halving the passes halves the cycles exactly.
A clean 2.00x here is the mechanism working, not a performance prediction.

What it leaves out is the activation feed. A packed pass consumes 2*size
activations per output row where dense consumes `size`, so the same bytes
arrive in half the time -- 2x the rate. The platform gives each scratchpad
pad one VLSU at one issue a cycle:

    DRAM -> Backend[p] -> scratchpad pad p -> VLSFrontendBridge -> VLSU p

so activations on a single pad deliver one vector a cycle, and a packed pass
wants two. On that wiring 2:4 is feed-bound well short of 2x.

The fix falls out of the weight side: a 2:4 tile is 2304 B against dense's
4096 B for the same K span (2048 B of survivors plus 256 B of indices), so
weights need 0.56x the space. Giving activations two pads and weights one
buys a second activation VLSU. What that cannot buy is DRAM bandwidth -- the
backends share one SharedDRAMBurstChannel, one burst launch a cycle in total
-- so a workload already DRAM-bound stays bound. Which limit binds is a
platform question; run it through sysarr_tpu_system to find out.
"""

import argparse
import csv
import sys

import numpy as np

from systolic_array.systolic_array_meissa import SystolicArrayMEISSA
from systolic_array.systolic_array_meissa_sparse import (
    SystolicArrayMEISSASparse24, prune_2to4)

SHAPES = [(32, 128), (32, 256), (64, 512), (32, 1024), (64, 2304)]


def _run_pass(sa, rows):
    cyc, sent = 0, 0
    limit = len(rows) + sa.flush_cycles() + sa.size + 8
    for _ in range(limit):
        if sent < len(rows) and sa.enqueue(list(rows[sent])):
            sent += 1
        sa.tick(float(cyc))
        cyc += 1
    return sa.get_buffer(), cyc


def _gemm(cls, a, w, m, k, n, packed):
    """Run a tiled GEMM; returns (result, cycles, utilisation)."""
    step = 2 * n if packed else n
    acc = np.zeros((m, n), dtype=np.float32)
    cycles = act_sum = window = 0
    for k0 in range(0, k, step):
        sa = cls(size=n, dtype="bf16", collect_stats=True)
        if packed:
            sa.load_weights_2to4(w[k0:k0 + step, :])
        else:
            sa.load_weights(w[k0:k0 + step, :].tolist())
        cycles += n                                   # the weight load's pushes
        out, c = _run_pass(sa, a[:, k0:k0 + step])
        cycles += c
        act_sum += sa.active_pe_sum
        window += sa.compute_window_cycles
        acc += np.array(out[:m], dtype=np.float32)
    util = act_sum / max(window * n * n, 1)
    return acc, cycles, util


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=32, help="array dimension")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--csv", help="write the rows here as well")
    args = ap.parse_args(argv)

    n = args.size
    rng = np.random.default_rng(args.seed)
    rows = []

    print(f"{n}x{n} MEISSA, bf16. The same 2:4-pruned weights through both.\n")
    print(f"{'shape (M,K,N)':<18}{'arch':>12}{'passes':>8}{'cycles':>9}"
          f"{'PE util':>10}{'speedup':>9}{'max err':>9}")
    print("-" * 76)

    for m, k in SHAPES:
        if k % (2 * n):
            continue
        a = rng.uniform(-2, 2, (m, k)).astype(np.float32)
        w = prune_2to4(rng.uniform(-2, 2, (k, n)).astype(np.float32))
        ref = a @ w
        scale = float(np.abs(ref).mean())

        got_d, cyc_d, u_d = _gemm(SystolicArrayMEISSA, a, w, m, k, n, False)
        got_s, cyc_s, u_s = _gemm(SystolicArrayMEISSASparse24, a, w, m, k, n, True)
        err_d = float(np.abs(got_d - ref).max() / scale)
        err_s = float(np.abs(got_s - ref).max() / scale)

        shape = f"({m},{k},{n})"
        print(f"{shape:<18}{'regular':>12}{k // n:>8}{cyc_d:>9}"
              f"{u_d:>9.1%}{1.0:>8.2f}x{err_d:>8.2%}")
        print(f"{'':<18}{'2:4 sparse':>12}{k // (2 * n):>8}{cyc_s:>9}"
              f"{u_s:>9.1%}{cyc_d / cyc_s:>8.2f}x{err_s:>8.2%}")
        print()
        rows.append(dict(M=m, K=k, N=n,
                         passes_regular=k // n, passes_sparse=k // (2 * n),
                         cycles_regular=cyc_d, cycles_sparse=cyc_s,
                         util_regular=round(u_d, 4), util_sparse=round(u_s, 4),
                         speedup=round(cyc_d / cyc_s, 4),
                         err_regular=round(err_d, 5), err_sparse=round(err_s, 5)))

    print("The speedup is the pass count: a packed pass costs what a dense one")
    print("costs, so this is the array in isolation. The activation feed is not")
    print("modelled here -- see the module docstring for why that matters.")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
