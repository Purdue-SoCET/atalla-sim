#!/usr/bin/env python3
"""Watch dense vs 2:4 array cycles build up, layer by layer, while a model runs.

    python tools/live_sparsity.py resnet_small_mnist_2to4
    python tools/live_sparsity.py resnet_small_mnist_2to4 --kernel-objs DIR --csv live.csv
    python tools/live_sparsity.py resnet_small_mnist_2to4 --from-trace trace.npz   # replay, fast

The model runs on atalla-sim as usual (dense MEISSA). Every array pass it
makes is caught as it finishes and run, cycle by cycle, through a standalone
dense MEISSA and -- once a layer's run of K tiles is complete -- through the
2:4 MEISSA as packed passes (each adjacent pair of 32-wide K tiles becomes one
64-wide pass). The table shows, per layer:

    passes      dense -> 2:4
    dense cyc   array cycles on regular MEISSA (32 weight pushes + compute + drain)
    2:4 cyc     array cycles on 2:4 MEISSA (32 pushes + 4 for indices + compute + drain)
    speedup     dense / 2:4
    err         worst |sparse - dense| / mean |dense| over the layer's packed passes

and the sim's own progress: cycles, rate, and the share of cycles the array
was busy. The array cycles are the array fed at full rate; the sim cycles are
what today's kernels actually take (see tools/replay_sparse.py).

Not a TTY (piped, nohup)? It prints one line per finished layer instead of
redrawing. The PyTorch check of every layer runs at the end, as in verify.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "models", ROOT / "models" / "atalla-graph", ROOT / "tools"):
    sys.path.insert(0, str(p))

from systolic_array.systolic_array_meissa import SystolicArrayMEISSA          # noqa: E402
from systolic_array.systolic_array_meissa_sparse import SystolicArrayMEISSASparse24  # noqa: E402

N = 32
META = (N * N * 2 // 8) // (N * 2)        # 256 B of indices at 64 B per push -> 4 cycles


def _drive(sa, acts):
    cyc = sent = 0
    while True:
        if sent < len(acts) and sa.enqueue(list(acts[sent])):
            sent += 1
        sa.tick(float(cyc))
        cyc += 1
        if sent == len(acts) and len(sa.get_buffer()) >= len(acts) and sa._is_drained():
            return np.array(sa.get_buffer()[:len(acts)], np.float32), cyc


class Layer:
    def __init__(self, plan):
        self.__dict__.update(plan)
        self.done_passes = 0
        self.sparse_passes = 0
        self.dense_cyc = 0
        self.closed_dense = 0             # dense cycles of K groups already compared
        self.sparse_cyc = 0
        self.worst_err = 0.0
        self.is24 = None                  # decided when its first K group completes
        self.group = []                   # (W, A, dense_out) of the K group in progress
        self.span = 0                     # sim cycles the array spent on this layer's passes

    @property
    def state(self):
        return "done" if self.done_passes == self.passes else ("running" if self.done_passes else "")


class Tracker:
    """Takes passes in schedule order, assigns them to layers, replays them."""

    def __init__(self, plan):
        self.layers = [Layer(p) for p in plan]
        self.cur = 0
        self.total_passes = sum(L.passes for L in self.layers)

    def add(self, W, A, span=0):
        L = self.layers[self.cur]
        sa = SystolicArrayMEISSA(size=N, dtype="bf16", use_native=False)
        sa.load_weights(W.tolist())
        out, c = _drive(sa, A)
        L.dense_cyc += N + c
        L.span += span
        L.done_passes += 1
        L.group.append((W, A, out, N + c))
        if len(L.group) == L.k_tiles:
            self._close_group(L)
        if L.done_passes == L.passes:
            self.cur += 1
            return L
        return None

    def _close_group(self, L):
        grp, L.group = L.group, []
        L.closed_dense += sum(g[3] for g in grp)
        if L.is24 is None:
            L.is24 = (L.kind == "conv" and L.k_tiles > 1 and
                      all(np.all((g[0].reshape(-1, 4, N) != 0).sum(1) <= 2) for g in grp))
        if not L.is24:
            L.sparse_passes += len(grp)
            L.sparse_cyc += sum(g[3] for g in grp)
            return
        for p in range(0, len(grp), 2):
            pair = grp[p:p + 2]
            (w0, a0, o0, _) = pair[0]
            if len(pair) == 2:
                (w1, a1, o1, _) = pair[1]
                w64, a64, ref = np.vstack([w0, w1]), np.hstack([a0, a1]), o0 + o1
            else:
                w64 = np.vstack([w0, np.zeros((N, N), np.float32)])
                a64, ref = np.hstack([a0, np.zeros_like(a0)]), o0
            sa = SystolicArrayMEISSASparse24(size=N, dtype="bf16")
            sa.load_weights_2to4(w64)
            out, c = _drive(sa, a64)
            L.sparse_cyc += N + META + c
            L.sparse_passes += 1
            scale = float(np.abs(ref).mean()) or 1.0
            L.worst_err = max(L.worst_err, float(np.abs(out - ref).max()) / scale)

    def totals(self, only24=False):
        """(dense, 2:4) array cycles over the K groups compared so far."""
        Ls = [L for L in self.layers if L.done_passes and (L.is24 or not only24)]
        return sum(L.closed_dense for L in Ls), sum(L.sparse_cyc for L in Ls)


def render(tr, sim_cycles, wall, array_busy, final=False):
    lines = []
    hdr = f"{'layer':<10}{'M':>5}{'K':>6}{'N':>5}  {'passes':>10}{'dense cyc':>11}{'2:4 cyc':>10}{'speedup':>9}{'err':>8}  state"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for L in tr.layers:
        if L.done_passes == 0:
            lines.append(f"{L.node:<10}{L.M:>5}{L.K:>6}{L.N:>5}  {'':>10}{'':>11}{'':>10}{'':>9}{'':>8}  ")
            continue
        sp = L.sparse_passes if L.is24 else L.done_passes
        passes = f"{L.done_passes}->{sp}" if L.is24 else f"{L.done_passes} dense"
        ratio = L.closed_dense / L.sparse_cyc if L.sparse_cyc else float("nan")
        err = f"{L.worst_err:.1%}" if L.is24 else ""
        lines.append(f"{L.node:<10}{L.M:>5}{L.K:>6}{L.N:>5}  {passes:>10}{L.dense_cyc:>11,}{L.sparse_cyc:>10,}"
                     f"{ratio:>8.2f}x{err:>8}  {L.state}")
    lines.append("-" * len(hdr))
    d24, s24 = tr.totals(only24=True)
    dall, sall = tr.totals()
    if s24:
        lines.append(f"2:4 layers so far   {d24:>9,} -> {s24:>9,} array cycles   {d24 / s24:.2f}x")
    if sall:
        lines.append(f"all layers so far   {dall:>9,} -> {sall:>9,} array cycles   {dall / sall:.2f}x")
    done = sum(L.done_passes for L in tr.layers)
    rate = sim_cycles / wall if wall else 0
    busy = f"{array_busy / sim_cycles:.1%}" if sim_cycles else "-"
    lines.append(f"sim {sim_cycles:,} cycles  {rate:,.0f} cyc/s  {wall / 60:.1f} min   "
                 f"passes {done}/{tr.total_passes}   array busy {busy} of sim cycles"
                 + ("   [finished]" if final else ""))
    return "\n".join(lines)


def write_csv(tr, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["layer", "M", "K", "N", "dense_passes", "sparse_passes", "dense_array_cycles",
                    "sparse_array_cycles", "speedup", "worst_err", "sim_array_span"])
        for L in tr.layers:
            sp = L.sparse_passes if L.is24 else L.done_passes
            w.writerow([L.node, L.M, L.K, L.N, L.done_passes, sp, L.dense_cyc, L.sparse_cyc,
                        round(L.dense_cyc / L.sparse_cyc, 4) if L.sparse_cyc else "",
                        round(L.worst_err, 5), L.span])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", default="resnet_small_mnist_2to4")
    ap.add_argument("--out-dir", type=Path, help="build directory (default models/out/MODEL)")
    ap.add_argument("--kernel-objs", type=Path, help="reuse compiled objects from this earlier build")
    ap.add_argument("--from-trace", type=Path, help="replay a trace_array_passes.py trace instead of simulating")
    ap.add_argument("--csv", type=Path, help="write the per-layer table here at the end")
    ap.add_argument("--refresh", type=float, default=2.0, help="seconds between redraws (default 2)")
    ap.add_argument("--limit", type=int, default=100_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=0.01)
    args = ap.parse_args()

    from replay_sparse import layer_plan
    tr = Tracker(layer_plan(args.model))
    tty = sys.stdout.isatty()

    def show(sim_cycles, wall, busy, final=False):
        text = render(tr, sim_cycles, wall, busy, final)
        if tty:
            sys.stdout.write("\033[H\033[J" + text + "\n")
            sys.stdout.flush()
        elif final:
            print(text, flush=True)

    def finished_layer(L, sim_cycles, wall):
        if not tty and L is not None:
            ratio = L.dense_cyc / L.sparse_cyc if L.sparse_cyc else float("nan")
            tag = f"{L.passes}->{L.sparse_passes}" if L.is24 else f"{L.passes} dense"
            print(f"[{wall / 60:5.1f} min, sim {sim_cycles:>12,}] {L.node:<9} K={L.K:<5} {tag:>10}  "
                  f"array {L.dense_cyc:,} -> {L.sparse_cyc:,} cycles  {ratio:.2f}x", flush=True)

    started = time.monotonic()
    if args.from_trace:
        t = np.load(args.from_trace)
        W, A, rows, st = t["W"], t["A"], t["rows"], t["stamps"]
        starts = np.concatenate([[0], np.cumsum(rows)])
        busy = 0
        for j in range(len(W)):
            span = int(st[j, 4] - st[j, 0] + 1)
            busy += span
            L = tr.add(W[j], A[starts[j]:starts[j + 1]], span)
            wall = time.monotonic() - started
            finished_layer(L, int(st[j, 4]), wall)
            if tty:
                show(int(st[j, 4]), wall, busy)
        show(int(st[-1, 4]), time.monotonic() - started, busy, final=True)
    else:
        import build_model
        from trace_array_passes import PassRecorder
        from scheduler.elf import load_elf
        from atalla.atalla_platform import build_atalla_platform, c_entry_sregs

        out_dir = args.out_dir or ROOT / "models" / "out" / args.model
        elf = build_model.build(args.model, out_dir, args.seed, args.scale, args.kernel_objs)
        program = load_elf(elf)
        platform = build_atalla_platform(program.instr, program.data, init_sregs=c_entry_sregs(program.data),
                                         describe_pc=program.describe, strict=True)
        rec = PassRecorder(platform, program.describe)
        fed, busy, last_draw = 0, 0, 0.0

        def feed(upto):
            nonlocal fed, busy
            while fed < upto:
                ps = rec.passes[fed]
                pushes = np.asarray(ps["pushes"], np.float32)
                span = int(ps["a1"] - ps["w0"] + 1) if ps["a1"] is not None else 0
                busy += span
                L = tr.add(pushes[::-1].T.copy(), np.asarray(ps["acts"], np.float32), span)
                finished_layer(L, platform.cycle, time.monotonic() - started)
                fed += 1

        while not platform.core.done:
            if platform.cycle >= args.limit:
                raise RuntimeError(f"No halt after {args.limit} cycles")
            platform.tick()
            if platform.cycle % 4096 == 0:
                feed(len(rec.passes))          # closed passes only; the open one is still loading
                now = time.monotonic()
                if tty and now - last_draw >= args.refresh:
                    show(platform.cycle, now - started, busy)
                    last_draw = now
        rec.finish()
        feed(len(rec.passes))
        show(platform.cycle, time.monotonic() - started, busy, final=True)

        import verify_model
        from validation_metrics import compare_cosine_max_mean
        worst = 1.0
        for name, op, info, ref in verify_model.expected_results(args.model, out_dir, args.seed, args.scale):
            actual = verify_model.read_tensor(platform, info)
            cos, mx, _ = compare_cosine_max_mean(actual, ref)
            worst = min(worst, cos)
        logits = verify_model.read_tensor(platform, info)
        print(f"PyTorch check: lowest per-layer cosine similarity {worst:.6f}; "
              f"predicted class {int(np.asarray(logits).reshape(-1).argmax())}")

    if args.csv:
        write_csv(tr, args.csv)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
