#!/usr/bin/env python3
"""Every INT8 scheme tried so far, plus the float formats, on the multi-format
MEISSA datapath -- with each INT8 output path:

  int32       32-bit lanes: the exact int sum leaves the array; K tiles add
              exactly (or in FP32 when the scale changes from pass to pass)
  bf16/bf16   16-bit lanes: int -> FP32 (exact) -> existing BF16 reducer;
              K tiles add in BF16 on the vector core
  bf16/fp32   same reducer, K tiles add in FP32 (what a wider accumulator buys)
  fp16/fp16   int -> FP32 -> existing FP16 reducer, with a 2^-4 exponent offset
              so a 32-term sum (up to 2^19) fits; K tiles add in FP16

Float formats (inputs rounded to the format, exact products, FP32 tree, the
RTL's reducer: BF16 -> BF16, FP16 / E5M2 -> FP16), K tiles added in the same
16-bit format or in FP32.

One array pass = 32 K terms. Per pass the model computes exactly what the
array outputs; `verify()` checks that against atalla-sim's cycle model bit for
bit. The vector core then adds passes in order, rounding after every add.
Scales: constant over K (per-tensor, per-row/col) are applied once at the end;
ones that change per pass (per-32-block, multi-term schemes) are applied to
each pass before it is added, rounded to the accumulator format.

Reference: the FP64 product of the original (unquantized) data.

    tools/int8_accuracy_sweep.py                  # 1024^3, writes logs/int8_accuracy/
    tools/int8_accuracy_sweep.py --size 256 --seeds 1 --no-k-sweep   # quick look
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from systolic_array.systolic_array_meissa import SystolicArrayMEISSA

T = 32
FMTS = {"fp16": (11, -14, 65504.0), "bf16": (8, -126, 3.3895313892515355e38),
        "e5m2": (3, -14, 57344.0)}
INT_PATHS = {            # name: (reducer format or None, exponent offset, acc)
    "int32": (None, 0, "exact"),
    "bf16/bf16": ("bf16", 0, "bf16"),
    "bf16/fp32": ("bf16", 0, "fp32"),
    "fp16/fp16": ("fp16", -4, "fp16"),
}


def rnd(x, fmt):
    """Round to nearest even in `fmt`; overflow goes to inf, as the RTL does."""
    x = np.asarray(x, dtype=np.float64)
    if fmt in ("exact", "fp64"):
        return x
    if fmt == "fp32":
        return x.astype(np.float32).astype(np.float64)
    p, emin, mx = FMTS[fmt]
    _, e = np.frexp(x)                          # |x| = m 2^e, m in [0.5, 1)
    step = np.exp2(np.maximum(e - 1, emin) - (p - 1))
    r = np.rint(x / step) * step
    with np.errstate(invalid="ignore"):
        big = np.abs(r) > mx
    if fmt == "e5m2":                           # saturating, as a quantizer
        return np.where(big, np.sign(r) * mx, r)
    return np.where(big, np.sign(r) * np.inf, r)


# --- what one array pass outputs ----------------------------------------------
def tree32(a, w):
    """Float pass: exact products, pairwise FP32 tree, as _reduce_columns."""
    lvl = (a[:, :, None].astype(np.float32) * w[None, :, :].astype(np.float32))
    while lvl.shape[1] > 1:
        lvl = lvl[:, 0::2] + lvl[:, 1::2]
    return lvl[:, 0].astype(np.float64)


def float_pass(a, w, fmt):
    return rnd(tree32(a, w), "bf16" if fmt == "bf16" else "fp16")


def int_pass_out(P, out, off):
    return P if out is None else rnd(np.ldexp(P, off), out)


# --- quantizers ---------------------------------------------------------------
def q(x, axis, pow2=False):
    s = np.max(np.abs(x), axis=axis, keepdims=axis is not None) / 127.0
    if pow2:
        s = np.exp2(np.ceil(np.log2(s)))
    return np.clip(np.rint(x / s), -127, 127), s


def hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def ozaki_slices(x, axis, s):
    e = np.ceil(np.log2(np.max(np.abs(x), axis=axis, keepdims=True) / 127))
    r = x / np.exp2(e)
    qs = []
    for _ in range(s):
        qq = np.clip(np.rint(r), -128, 127)
        qs.append(qq)
        r = (r - qq) * 128
    return qs, np.exp2(e)


# A scheme is a list of int terms (qa, qw, scale(k0) -> M x N or scalar,
# per_pass) plus optional float terms (a, w, fmt). per_pass: the scale differs
# between passes or terms, so it is applied before accumulating.
def schemes(A, W, idx):
    M, N = A.shape[0], W.shape[1]
    out = {}
    qa, sa = q(A, None); qw, sw = q(W, None)
    out["per-tensor"] = dict(terms=[(qa, qw, lambda k, s=sa * sw: s, False)])
    qa, sa = q(A, 1); qw, sw = q(W, 0)
    rc = (qa, qw, sa, sw)
    out["per-row/col"] = dict(terms=[(qa, qw, lambda k, s=sa * sw: s, False)])
    for name, pow2 in (("per-32-block", False), ("MXINT8 (pow2 per-32-block)", True)):
        QA, QW, SA, SW = np.empty_like(A), np.empty_like(W), {}, {}
        for k0 in range(0, A.shape[1], T):
            ks = slice(k0, k0 + T)
            QA[:, ks], SA[k0] = q(A[:, ks], 1, pow2)
            QW[ks], SW[k0] = q(W[ks], 0, pow2)
        out[name] = dict(terms=[(QA, QW, (lambda SA, SW: lambda k: SA[k] * SW[k])(SA, SW), True)])
    # hi/lo split: x ~= q s + r t, three INT8 GEMMs
    qa, qw, sa, sw = rc
    ra, ta = q(A - qa * sa, 1); rw, tw = q(W - qw * sw, 0)
    out["hi/lo split (3 passes)"] = dict(terms=[
        (qa, qw, lambda k, s=sa * sw: s, True), (qa, rw, lambda k, s=sa * tw: s, True),
        (ra, qw, lambda k, s=ta * sw: s, True)])
    for s in (2, 3, 4):
        QA, ea = ozaki_slices(A, 1, s); QW, ew = ozaki_slices(W, 0, s)
        terms = [(QA[i], QW[j], (lambda i, j: lambda k: ea * ew * 2.0 ** (-7 * (i + j)))(i, j), True)
                 for i in range(s) for j in range(s) if i + j < s]
        out[f"Ozaki s={s} ({len(terms)} passes)"] = dict(terms=terms)
    if idx is not None:
        keep = np.setdiff1d(np.arange(A.shape[1]), idx)
        qa, sa = q(A[:, keep], 1); qw, sw = q(W[keep], 0)
        out["LLM.int8() (outlier channels FP16)"] = dict(
            terms=[(qa, qw, lambda k, s=sa * sw: s, True)],
            float_terms=[(A[:, idx], W[idx], "fp16")])
        sm = np.max(np.abs(A), 0) ** 0.5 / np.max(np.abs(W), 1) ** 0.5
        qa, sa = q(A / sm, 1); qw, sw = q(W * sm[:, None], 0)
        out["SmoothQuant (a=0.5)"] = dict(terms=[(qa, qw, lambda k, s=sa * sw: s, False)])
        H = hadamard(A.shape[1])
        qa, sa = q(A @ H, 1); qw, sw = q(H.T @ W, 0)
        out["Hadamard rotation"] = dict(terms=[(qa, qw, lambda k, s=sa * sw: s, False)])
    return out


def pad_k(x, axis):
    short = (-x.shape[axis]) % T
    if not short:
        return x
    shape = list(x.shape); shape[axis] = short
    return np.concatenate([x, np.zeros(shape)], axis=axis)


def run_int(sch):
    """All four INT8 output paths for one scheme, sharing the exact pass sums."""
    acc = {p: 0.0 for p in INT_PATHS}
    terms = sch["terms"]
    for (qa, qw, scale, per_pass) in terms:
        qa, qw = pad_k(qa, 1), pad_k(qw, 0)
        for k0 in range(0, qa.shape[1], T):
            P = qa[:, k0:k0 + T] @ qw[k0:k0 + T]        # exact: |P| < 2^21
            for p, (out, off, af) in INT_PATHS.items():
                v = int_pass_out(P, out, off)   # still carries 2^off
                if per_pass:
                    af2 = "fp32" if af == "exact" else af
                    c = rnd(v * rnd(scale(k0) * 2.0 ** -off, af2), af2)
                    acc[p] = rnd(acc[p] + c, af2)
                else:
                    acc[p] = rnd(acc[p] + v, af) if af != "exact" else acc[p] + v
    if not terms[0][3]:                                  # scale once at the end
        s = terms[0][2](0)
        for p, (_, off, af) in INT_PATHS.items():
            af2 = "fp32" if af == "exact" else af
            acc[p] = rnd(acc[p] * rnd(s * 2.0 ** -off, af2), af2)
    for (a, w, fmt) in sch.get("float_terms", ()):
        a, w = pad_k(rnd(a, fmt), 1), pad_k(rnd(w, fmt), 0)
        for k0 in range(0, a.shape[1], T):
            v = float_pass(a[:, k0:k0 + T], w[k0:k0 + T], fmt)
            for p, (_, _, af) in INT_PATHS.items():
                af2 = "fp32" if af == "exact" else af
                acc[p] = rnd(acc[p] + v, af2)
    return acc


def run_float(A, W, fmt, accs):
    a = rnd(A, fmt); w = rnd(W, fmt)
    acc = {af: 0.0 for af in accs}
    for k0 in range(0, A.shape[1], T):
        v = float_pass(a[:, k0:k0 + T], w[k0:k0 + T], fmt)
        for af in accs:
            acc[af] = rnd(acc[af] + v, af)
    return acc


def metrics(out, ref):
    err = np.abs(out - ref)
    rms = np.sqrt(np.mean(ref ** 2))
    mask = np.abs(ref) >= 1e-3 * rms
    rel = err[mask] / np.abs(ref[mask])
    ulps = err / np.spacing(np.abs(ref.astype(np.float16))).astype(np.float64)
    return {"rel_rms": float(np.sqrt(np.mean(err ** 2)) / rms),
            "mre": float(rel.mean()), "median_rel": float(np.median(rel)),
            "mean_ulp": float(np.mean(ulps)), "median_ulp": float(np.median(ulps)),
            "nonfinite": int((~np.isfinite(out)).sum())}


def data(kind, seed, M, K, N):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, K)); W = rng.standard_normal((K, N))
    idx = None
    if kind == "outliers":                # 0.5% of activation channels 50x larger
        idx = rng.choice(K, max(1, int(0.005 * K)), replace=False)
        A[:, idx] *= 50
    return A, W, idx


def job(args):
    kind, seed, M, K, N, which = args
    t0 = time.time()
    A, W, idx = data(kind, seed, M, K, N)
    ref = A @ W
    res = []
    if which in ("all", "float"):
        for fmt in ("fp16", "bf16", "e5m2"):
            own = "bf16" if fmt == "bf16" else "fp16"
            for af, o in run_float(A, W, fmt, (own, "fp32")).items():
                res.append(dict(group="float", scheme=fmt.upper(),
                                path=f"{own}/{af}", **metrics(o, ref)))
        # W8A16: INT8 weights dequantized to FP16 on load, FP16 activations,
        # through the FP16 path
        qw, sw = q(W, 0)
        wd = rnd(qw * sw, "fp16")
        a = rnd(A, "fp16")
        for af in ("fp16", "fp32"):
            accv = 0.0
            for k0 in range(0, K, T):
                accv = rnd(accv + float_pass(a[:, k0:k0 + T], wd[k0:k0 + T], "fp16"), af)
            res.append(dict(group="float", scheme="W8A16", path=f"fp16/{af}",
                            **metrics(accv, ref)))
    if which in ("all", "int"):
        for name, sch in schemes(A, W, idx).items():
            for p, o in run_int(sch).items():
                res.append(dict(group="int8", scheme=name, path=p, **metrics(o, ref)))
    for r in res:
        r.update(kind=kind, seed=seed, M=M, K=K, N=N)
    print(f"done {kind} seed={seed} K={K} {which} in {time.time() - t0:.0f}s", flush=True)
    return res


# --- bit-exact check against atalla-sim's cycle model --------------------------
def sim_pass(a, w, dtype, int_output=None, off=0):
    sa = SystolicArrayMEISSA(size=T, dtype=dtype, use_mixed_adder=False,
                             int_output=int_output, int_exp_offset=off)
    sa.load_weights(w.tolist())
    sent, cyc, M = 0, 0, a.shape[0]
    while len(sa.get_buffer()) < M:
        if sent < M and sa.enqueue(a[sent].tolist()):
            sent += 1
        sa.tick(float(cyc))
        cyc += 1
    assert sa.overflow_count == 0
    return np.array(sa.get_buffer(), dtype=np.float64)


def verify(M=256, seed=7):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, T)); W = rng.standard_normal((T, T))
    A[:, 3] *= 50
    bad = {}
    for fmt, dt in (("fp16", "fp16"), ("bf16", "bf16"), ("e5m2", "fp16")):
        a, w = rnd(A, fmt), rnd(W, fmt)
        bad[fmt] = int((sim_pass(a, w, dt) != float_pass(a, w, fmt)).sum())
    qa, _ = q(A, 1); qw, _ = q(W, 0)
    P = qa @ qw
    for p, (out, off, _) in INT_PATHS.items():
        if p == "bf16/fp32":
            continue
        got = sim_pass(qa, qw, "int8", out, off)
        bad["int8 " + p] = int((got != int_pass_out(P, out, off)).sum())
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--size", type=int, default=1024, help="M = K = N")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no-k-sweep", action="store_true",
                    help="skip the K = 32 .. size/2 runs on normal data")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--output", default=str(REPO_ROOT / "logs" / "int8_accuracy" / "results.json"))
    args = ap.parse_args()

    bad = verify()
    print("sim vs model mismatches (of", 256 * T, "outputs):", bad, flush=True)
    if any(bad.values()):
        sys.exit("the vectorized model no longer matches SystolicArrayMEISSA")
    S, seeds = args.size, range(args.seeds)
    jobs = [(k, s, S, S, S, "all") for k in ("normal", "outliers") for s in seeds]
    if not args.no_k_sweep:
        Ks = [K for K in (32, 128, 512) if K < S]
        jobs += [("normal", s, S, K, S, w) for K in Ks for s in seeds
                 for w in ("float", "int")]
    with Pool(args.jobs) as p:
        results = [r for rs in p.map(job, jobs, chunksize=1) for r in rs]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    np.seterr(over="ignore", invalid="ignore")
    main()
