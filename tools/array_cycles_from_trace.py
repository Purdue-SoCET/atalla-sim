#!/usr/bin/env python3
"""Measured array cycles, dense MEISSA vs 2:4 MEISSA, on a recorded model run.

    python tools/array_cycles_from_trace.py TRACE.npz --model resnet_small_mnist_2to4

Every recorded pass is run cycle by cycle through a standalone SystolicArrayMEISSA
(N cycles of weight pushes + compute until the last output drains). For layers
that are 2:4 with more than one K tile, each adjacent pair of K tiles is run as
one packed pass through SystolicArrayMEISSASparse24 (N pushes + 4 cycles of
index metadata + compute). An odd last K tile runs as a half-full packed pass.

This is the array alone, fed one push or row per cycle. On today's platform the
CPU issues them far slower than that; tools/replay_sparse.py shows the trace's
own timing.
"""
import argparse, math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "models", ROOT / "models" / "atalla-graph", ROOT / "tools"):
    sys.path.insert(0, str(p))
from systolic_array.systolic_array_meissa import SystolicArrayMEISSA
from systolic_array.systolic_array_meissa_sparse import SystolicArrayMEISSASparse24
from replay_sparse import layer_plan
N = 32
META = (N * N * 2 // 8) // (N * 2)        # 256 B of indices at 64 B per push


def drive(sa, acts):
    cyc = sent = 0
    while True:
        if sent < len(acts) and sa.enqueue(list(acts[sent])):
            sent += 1
        sa.tick(float(cyc)); cyc += 1
        if sent == len(acts) and len(sa.get_buffer()) >= len(acts) and sa._is_drained():
            return cyc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", type=Path); ap.add_argument("--model", required=True)
    args = ap.parse_args()
    t = np.load(args.trace)
    W, A, rows = t["W"], t["A"], t["rows"]
    st = np.concatenate([[0], np.cumsum(rows)])
    i = 0; td = ts = sd = ss = 0
    for L in layer_plan(args.model):
        idx = list(range(i, i + L["passes"])); i += L["passes"]; kt = L["k_tiles"]
        d = 0
        for j in idx:
            sa = SystolicArrayMEISSA(size=N, dtype="bf16", use_native=False); sa.load_weights(W[j].tolist())
            d += N + drive(sa, A[st[j]:st[j + 1]])
        is24 = L["kind"] == "conv" and kt > 1 and all(
            np.all((W[j].reshape(-1, 4, N) != 0).sum(1) <= 2) for j in idx)
        s, merged = (0, 0) if is24 else (d, L["passes"])
        if is24:
            for g in range(0, len(idx), kt):
                grp = idx[g:g + kt]
                for p in range(0, kt, 2):
                    pr = grp[p:p + 2]; a0 = A[st[pr[0]]:st[pr[0] + 1]]
                    if len(pr) == 2:
                        w64 = np.vstack([W[pr[0]], W[pr[1]]]); a64 = np.hstack([a0, A[st[pr[1]]:st[pr[1] + 1]]])
                    else:
                        w64 = np.vstack([W[pr[0]], np.zeros((N, N), np.float32)]); a64 = np.hstack([a0, np.zeros_like(a0)])
                    sa = SystolicArrayMEISSASparse24(size=N, dtype="bf16"); sa.load_weights_2to4(w64)
                    s += N + META + drive(sa, a64); merged += 1
            sd += d; ss += s
        td += d; ts += s
        print(f"{L['node']:<10} M={L['M']:<4} K={L['K']:<5} N={L['N']:<4} passes {L['passes']:>3} -> {merged:>3}"
              f"   array cycles {d:>7,} -> {s:>7,}   {d / s:.2f}x")
    print(f"\n2:4 layers: {sd:,} -> {ss:,} ({sd / ss:.2f}x)   whole array: {td:,} -> {ts:,} ({td / ts:.2f}x)")


if __name__ == "__main__":
    main()
