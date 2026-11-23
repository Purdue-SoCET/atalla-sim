#!/usr/bin/env python3
"""gelu.vl's arithmetic vs exact GELU, over every BF16 value in a range."""
import math, sys
from pathlib import Path
import numpy as np

FSIM = Path(__file__).resolve().parents[1] / "third_party/atalla-functional-sim"
sys.path.insert(0, str(FSIM))
from src.components.vector_loop import gelu          # the sim's BF16 pipeline

# Every BF16 bit pattern -> float32, keep finite values in [-LIM, LIM].
LIM = 8.0
bits = np.arange(1 << 16, dtype=np.uint32) << 16
x = bits.view(np.float32)
x = x[np.isfinite(x) & (np.abs(x) <= LIM)]

erf = np.vectorize(math.erf)
xd = x.astype(np.float64)
k = math.sqrt(2.0 / math.pi)
tanh_gelu = 0.5 * xd * (1.0 + np.tanh(k * (xd + 0.044715 * xd**3)))   # the target, float64
exact = 0.5 * xd * (1.0 + erf(xd / math.sqrt(2.0)))   # true GELU, float64
approx = xd / (1.0 + np.exp(1.702 * -xd))             # sigmoid approximation, float64, no rounding
sim = gelu(x).astype(np.float64)                      # what the hardware computes

def report(name, got, ref):
    err = np.abs(got - ref)
    i = int(np.argmax(err))
    big = np.abs(ref) > 1e-2
    rel = np.max(err[big] / np.abs(ref[big]))
    print(f"{name:28s} max abs {err.max():.4g} at x={x[i]:+.4f}   "
          f"mean abs {err.mean():.3g}   max rel (|y|>0.01) {rel:.3g}")

report("BF16 pipeline vs tanh-GELU", sim, tanh_gelu)   # what gelu.vl must match
report("tanh-GELU vs exact (erf-GELU)", tanh_gelu, exact)      # for context only
report("approximation vs exact (erf-GELU)", approx, exact)   # the formula's own error
report("BF16 pipeline vs approx", sim, approx)    # rounding error alone
report("BF16 pipeline vs exact (erf-GELU)", sim, exact)      # both together

# In BF16 ULPs: how far from the exactly rounded true answer.
exact_bf = np.asarray(exact, np.float32).view(np.uint32) + 0x7FFF + ((np.asarray(exact, np.float32).view(np.uint32) >> 16) & 1)
exact_bf = (exact_bf & 0xFFFF0000).view(np.float32)
ulp = np.abs(sim.astype(np.float32).view(np.int32) - exact_bf.view(np.int32)) >> 16
print("ULPs from exact rounded:", {int(k): int(v) for k, v in zip(*np.unique(ulp, return_counts=True))})
