#!/usr/bin/env python3
"""Inputs and PyTorch's output for examples/model/mlp.plan.json.

Builds the atalla-graph example's MLP (Linear 4->8, ReLU, Linear 8->3) in
BF16 with a fixed seed, and writes

    mlp.inputs.npz   arg0 = x (2x4), arg1/arg2 = layer 1 weight and bias,
                     arg3/arg4 = layer 2 weight and bias: the plan's inputs, in order
    mlp.torch.npy    PyTorch's BF16 output for x

    python examples/model/export_mlp.py -o examples/model     # needs torch
    atalla-sim model examples/model/mlp.plan.json --inputs examples/model/mlp.inputs.npz \\
        --reference examples/model/mlp.torch.npy
"""
import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-o", "--out", type=Path, default=Path("."))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(),
                                torch.nn.Linear(8, 3)).to(torch.bfloat16).eval()
    x = torch.randn(2, 4).to(torch.bfloat16)
    with torch.no_grad():
        y = model(x)
    f32 = lambda t: t.detach().to(torch.float32).numpy()  # noqa: E731
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "mlp.inputs.npz", arg0=f32(x), arg1=f32(model[0].weight),
             arg2=f32(model[0].bias), arg3=f32(model[2].weight), arg4=f32(model[2].bias))
    np.save(args.out / "mlp.torch.npy", f32(y))
    print("wrote %s and %s" % (args.out / "mlp.inputs.npz", args.out / "mlp.torch.npy"))


if __name__ == "__main__":
    main()
