#!/usr/bin/env python3
"""Replay a recorded model run through the 2:4 sparse MEISSA array and
estimate what the whole model would take on it.

    python tools/replay_sparse.py TRACE.npz --model resnet_small_mnist_2to4 [--csv out.csv]

TRACE comes from tools/trace_array_passes.py. The conv kernel walks K tiles
innermost (spatial.c: n_tiles -> position blocks -> ki), so a layer's passes
come in runs of k_tiles that all feed one output tile. On the sparse array
each adjacent pair of those -- two 32-wide K slices -- becomes ONE packed
pass over 64 of K; an odd last tile runs as a packed pass half full.

CORRECTNESS. Every merged pass is run through SystolicArrayMEISSASparse24
with the recorded weights (packing refuses anything that is not 2:4) and
the recorded activations, and compared with what the dense array actually
returned for the same K span (its pass outputs summed, as the vector core
accumulates them) and with an fp32 reference.

CYCLES. The trace stamps every pass: weight load (w0..w1), activation
stream (a0..a1), last output (o1). A merged pair still needs both K slices
gathered by im2col, so all CPU work between passes is kept. What it drops
is the second pass. Two bounds, because the platform as wired cannot feed a
64-wide activation row in the time it feeds a 32-wide one:

  feed-bound   (today's wiring, one activation VLSU): the second pass's
               weight load is saved, its activation stream is not.
  array-bound  (activations split across two pads): the second pass's
               weight load AND its activation stream are saved.

Both add the 2-bit index load (1/8 of a weight load) back per merged pass.
Layers that stay dense (stem, FC) cost what they cost.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "models", ROOT / "models" / "atalla-graph"):
    sys.path.insert(0, str(p))

from systolic_array.systolic_array_meissa_sparse import SystolicArrayMEISSASparse24  # noqa: E402

N = 32


def _drive(sa, rows):
    cyc = sent = 0
    for _ in range(len(rows) + sa.flush_cycles() + sa.size + 8):
        if sent < len(rows) and sa.enqueue(list(rows[sent])):
            sent += 1
        sa.tick(float(cyc))
        cyc += 1
    return np.array(sa.get_buffer()[:len(rows)], np.float32)


def _bf16(x):
    x = np.asarray(x, np.float32)
    b = x.view(np.uint32)
    b = ((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return b.view(np.float32)


def layer_plan(model_name: str):
    """(node, kind, M, K, N) for every layer that uses the array, in schedule
    order, from the traced graph."""
    import torch
    from run_schedule import build_graph, load_model
    torch.manual_seed(42)
    model, x = load_model(model_name)
    graph = build_graph(model, x)
    out = []
    for node in graph.graph.nodes:
        if node.op != "call_function":
            continue
        name = getattr(node.target, "__name__", str(node.target))
        if name == "conv2d":
            wshape = node.args[1].meta["tensor_meta"].shape
            oshape = node.meta["tensor_meta"].shape
            out.append(dict(node=node.name, kind="conv", M=oshape[-1] * oshape[-2],
                            K=wshape[1] * wshape[2] * wshape[3], N=wshape[0]))
        elif name == "matmul":
            a, b = (arg.meta["tensor_meta"].shape for arg in node.args[:2])
            out.append(dict(node=node.name, kind="matmul", M=a[-2], K=a[-1], N=b[-1]))
    for L in out:
        L["k_tiles"] = math.ceil(L["K"] / N)
        L["passes"] = math.ceil(L["M"] / N) * math.ceil(L["N"] / N) * L["k_tiles"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--total-cycles", type=int, help="the run's total cycles (default: from trace_report.json)")
    ap.add_argument("--report", type=Path, help="trace_report.json of the run")
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()

    t = np.load(args.trace)
    W, A, O, rows, st = t["W"], t["A"], t["O"], t["rows"], t["stamps"]
    starts = np.concatenate([[0], np.cumsum(rows)])
    plan = layer_plan(args.model)
    assert sum(L["passes"] for L in plan) == len(W), \
        f"plan has {sum(L['passes'] for L in plan)} passes, trace has {len(W)}"

    load = st[:, 1] - st[:, 0] + 1          # weight-load phase
    stream = st[:, 3] - st[:, 2] + 1        # activation-stream phase
    total = args.total_cycles
    if total is None and args.report:
        import json
        total = json.loads(args.report.read_text())["cycles"]

    rows_out, i = [], 0
    sum_fb = sum_ab = 0
    worst_sparse = worst_dense = 0.0
    for L in plan:
        idx = list(range(i, i + L["passes"]))
        i += L["passes"]
        Lw = W[idx]
        is_24 = L["kind"] == "conv" and all(
            np.all((Lw[j].reshape(-1, 4, N) != 0).sum(axis=1) <= 2) for j in range(len(idx)))
        L["sparse"] = bool(is_24 and L["k_tiles"] > 1)
        L["array_cycles"] = int(sum(st[j, 4] - st[j, 0] + 1 for j in idx))
        L["load_cycles"] = int(load[idx].sum())
        L["stream_cycles"] = int(stream[idx].sum())
        fb = ab = 0
        merged = 0
        if L["sparse"]:
            kt = L["k_tiles"]
            for g in range(0, len(idx), kt):
                grp = idx[g:g + kt]
                for p in range(0, kt, 2):
                    pair = grp[p:p + 2]
                    a0 = A[starts[pair[0]]:starts[pair[0] + 1]]
                    if len(pair) == 2:
                        a1 = A[starts[pair[1]]:starts[pair[1] + 1]]
                        assert len(a0) == len(a1), "paired passes stream different row counts"
                        w64 = np.vstack([W[pair[0]], W[pair[1]]])
                        a64 = np.hstack([a0, a1])
                        dense_out = _bf16(O[starts[pair[0]]:starts[pair[0] + 1]]
                                          + O[starts[pair[1]]:starts[pair[1] + 1]])
                        q = pair[1]
                        fb += load[q] - load[q] // 8
                        ab += load[q] + stream[q] - load[q] // 8
                    else:                                   # odd last K tile, half-full packed pass
                        w64 = np.vstack([W[pair[0]], np.zeros((N, N), np.float32)])
                        a64 = np.hstack([a0, np.zeros_like(a0)])
                        dense_out = O[starts[pair[0]]:starts[pair[0] + 1]]
                        fb -= load[pair[0]] // 8
                        ab -= load[pair[0]] // 8
                    sa = SystolicArrayMEISSASparse24(size=N, dtype="bf16")
                    sa.load_weights_2to4(w64)
                    got = _drive(sa, a64)
                    ref = a64 @ w64
                    scale = float(np.abs(ref).mean()) or 1.0
                    worst_sparse = max(worst_sparse, float(np.abs(got - ref).max()) / scale)
                    worst_dense = max(worst_dense, float(np.abs(dense_out - ref).max()) / scale)
                    merged += 1
        L["sparse_passes"] = merged if L["sparse"] else L["passes"]
        L["saved_feed_bound"] = int(fb)
        L["saved_array_bound"] = int(ab)
        sum_fb += fb
        sum_ab += ab
        rows_out.append(L)

    print(f"{'layer':<10}{'M,K,N':>16}{'passes':>14}{'array cyc':>11}{'load':>8}{'stream':>8}"
          f"{'save(feed)':>12}{'save(array)':>12}")
    for L in rows_out:
        tag = f"{L['passes']}->{L['sparse_passes']}" if L["sparse"] else f"{L['passes']} dense"
        print(f"{L['node']:<10}{L['M']:>5},{L['K']:>5},{L['N']:>4}{tag:>14}{L['array_cycles']:>11,}"
              f"{L['load_cycles']:>8,}{L['stream_cycles']:>8,}{L['saved_feed_bound']:>12,}{L['saved_array_bound']:>12,}")
    arr = sum(L["array_cycles"] for L in rows_out)
    print(f"\nmerged passes checked: worst error vs fp32 -- sparse array {worst_sparse:.2%}, "
          f"dense array (two passes summed) {worst_dense:.2%} (both relative to mean |ref|)")
    if total:
        print(f"whole run {total:,} cycles; array passes span {arr:,} ({arr / total:.1%})")
        for name, s in (("feed-bound", sum_fb), ("array-bound", sum_ab)):
            print(f"  {name:<12} sparse est. {total - s:,} cycles  -> {total / (total - s):.3f}x")
    if args.csv:
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)


if __name__ == "__main__":
    main()
