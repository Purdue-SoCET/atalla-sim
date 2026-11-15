#!/usr/bin/env python3
"""Run a model ELF on atalla-sim, check it against PyTorch, and record every
systolic-array pass it makes, for replay through the 2:4 sparse array.

    python tools/trace_array_passes.py resnet_small_mnist_2to4 \
        --out-dir models/out/resnet_small_mnist_2to4 --kernel-objs DIR \
        --trace trace.npz

A pass is one weight load (32 column pushes) and the activation rows streamed
through it. For each pass the trace keeps the placed weight tile W (row i =
activation lane i, column j = output j), the activation rows, the output rows
the array returned, and cycle stamps:

    w0, w1   first and last weight push
    a0, a1   first and last activation row in
    o1       last output row out

tools/replay_sparse.py consumes it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "models", ROOT / "models" / "atalla-graph"):
    sys.path.insert(0, str(p))


class PassRecorder:
    def __init__(self, platform, describe):
        self.p, self.describe = platform, describe
        self.sa, self.bridge = platform.sa, platform.sysarr_bridge
        self.passes, self.cur = [], None
        sa, bridge = self.sa, self.bridge
        orig_w, orig_a, orig_rsp = sa.enqueue_weights, sa.enqueue, bridge._pack_rsp

        def enqueue_weights(weights, dtype=None, *, count_algo=True):
            ok = orig_w(weights, dtype, count_algo=count_algo)
            if ok:
                now = int(bridge.now)
                if self.cur is None or self.cur["acts"]:
                    self._close()
                    self.cur = dict(pushes=[], acts=[], outs=[], w0=now, w1=now, a0=None, a1=None, o1=None,
                                    where=describe(platform.core.pc) if describe else "")
                self.cur["pushes"].append([float(x) for x in weights])
                self.cur["w1"] = now
            return ok

        def enqueue(activations, dtype=None, *, count_algo=True):
            ok = orig_a(activations, dtype, count_algo=count_algo)
            if ok and count_algo and self.cur is not None:
                now = int(bridge.now)
                self.cur["acts"].append([float(x) for x in activations])
                self.cur["a0"] = now if self.cur["a0"] is None else self.cur["a0"]
                self.cur["a1"] = now
            return ok

        self.out_rows, self.out_t = [], []

        def pack_rsp(out_row, meta):
            self.out_rows.append([float(x) for x in out_row[: sa.size]])
            self.out_t.append(int(bridge.now))
            return orig_rsp(out_row, meta)

        sa.enqueue_weights, sa.enqueue, bridge._pack_rsp = enqueue_weights, enqueue, pack_rsp

    def _close(self):
        if self.cur is not None:
            self.passes.append(self.cur)
        self.cur = None

    def finish(self):
        self._close()
        # Outputs come back in issue order, one per activation row, and a
        # pass's last ones can land after the next pass has started loading.
        k = 0
        for ps in self.passes:
            n = len(ps["acts"])
            ps["outs"] = self.out_rows[k:k + n]
            ps["o1"] = self.out_t[k + n - 1] if n else None
            k += n
        assert k == len(self.out_rows), (k, len(self.out_rows))
        return self.passes

    def save(self, path: Path):
        n = self.sa.size
        P = self.passes
        W = np.zeros((len(P), n, n), np.float32)
        for i, ps in enumerate(P):
            pushes = np.asarray(ps["pushes"], np.float32)        # (n pushes, n lanes)
            assert pushes.shape == (n, n), pushes.shape
            # column j holds push n-1-j; push vector index = activation lane
            W[i] = pushes[::-1].T
        rows = np.array([len(ps["acts"]) for ps in P])
        A = np.concatenate([np.asarray(ps["acts"], np.float32) for ps in P])
        O = np.concatenate([np.asarray(ps["outs"], np.float32).reshape(-1, n) for ps in P])
        stamps = np.array([[ps["w0"], ps["w1"], ps["a0"], ps["a1"], ps["o1"] or -1] for ps in P])
        np.savez_compressed(path, W=W, rows=rows, A=A, O=O, stamps=stamps,
                            where=np.array([ps["where"] for ps in P]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--kernel-objs", type=Path)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=100_000_000)
    args = ap.parse_args()

    import build_model
    import verify_model
    from validation_metrics import compare_cosine_max_mean
    from scheduler.elf import load_elf
    from atalla.atalla_platform import build_atalla_platform, c_entry_sregs

    elf = build_model.build(args.model, args.out_dir, args.seed, args.scale, args.kernel_objs)
    program = load_elf(elf)
    platform = build_atalla_platform(program.instr, program.data, init_sregs=c_entry_sregs(program.data),
                                     describe_pc=program.describe, strict=True)
    rec = PassRecorder(platform, program.describe)
    started = time.monotonic()
    while not platform.core.done:
        if platform.cycle >= args.limit:
            raise RuntimeError(f"No halt after {args.limit} cycles")
        platform.tick()
        if platform.cycle % 1_000_000 == 0:
            print(f"Simulated {platform.cycle:,} cycles ({time.monotonic() - started:.0f}s), "
                  f"{len(rec.passes)} passes", flush=True)
    rec.finish()
    rec.save(args.trace)

    results = []
    for name, op, info, ref in verify_model.expected_results(args.model, args.out_dir, args.seed, args.scale):
        actual = verify_model.read_tensor(platform, info)
        cos, mx, mean = compare_cosine_max_mean(actual, ref)
        ok = bool(np.allclose(actual, ref, atol=.01, rtol=.03, equal_nan=True))
        results.append(dict(node=name, operation=op, passed=ok, cosine_similarity=cos, max_abs=mx, mean_abs=mean))
        print(f"{'PASS' if ok else 'FAIL'} {name}: cos {cos:.6f}; max abs {mx:.6g}")
    logits = verify_model.read_tensor(platform, json.loads((args.out_dir / "tensor_manifest.json").read_text())[results[-1]["node"]])
    report = dict(model=args.model, cycles=platform.cycle, passes=len(rec.passes),
                  elf_sha256=hashlib.sha256(elf.read_bytes()).hexdigest(),
                  passed=all(r["passed"] for r in results), results=results,
                  logits=np.asarray(logits).reshape(-1).tolist(),
                  predicted=int(np.asarray(logits).reshape(-1).argmax()))
    (args.out_dir / "trace_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{args.model}: {'PASS' if report['passed'] else 'FAIL'}; {platform.cycle:,} cycles; "
          f"{len(rec.passes)} array passes; predicted {report['predicted']}; trace -> {args.trace}")


if __name__ == "__main__":
    main()
