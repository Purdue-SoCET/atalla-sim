#!/usr/bin/env python3
"""Classify your own picture of a digit with the MNIST-trained resnet_small,
dense and 2:4, and optionally run it on the Atalla sim.

    python tools/classify_digit.py my_digit.png
    python tools/classify_digit.py my_digit.jpg --sim          # + the Atalla sim (~75 min)

Any image PIL opens works: a photo of a digit on paper, a phone screenshot, a
drawing from Paint. It is turned into what the model was trained on, the way
MNIST itself was made:

  1. greyscale; if the background is light (dark ink on paper) it is inverted,
     so the digit is white on black like MNIST
  2. contrast stretched, faint paper texture zeroed
  3. cropped to the digit, scaled so its longer side is 20 px, centred in a
     28x28 square by its centre of mass
  4. then the training pipeline: resized to 16x16, normalised, copied to 3 channels

The two pictures it saves next to the image (*_28x28.png, *_16x16.png) show
what the model actually sees -- check them if a prediction looks wrong.

--sim builds resnet_small_mnist_2to4 with this image as its input and runs
tools/live_sparsity.py on it: the per-layer dense vs 2:4 array cycles, then the
sim's own prediction checked against PyTorch.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "models" / "atalla-graph"
for p in (ROOT / "src", ROOT / "models", GRAPH):
    sys.path.insert(0, str(p))


def to_mnist(path: Path) -> np.ndarray:
    """Any image -> a 28x28 uint8 MNIST-style digit (white on black, centred)."""
    from PIL import Image
    a = np.asarray(Image.open(path).convert("L"), np.float32) / 255.0
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    if np.median(border) > 0.5:                      # light paper, dark ink
        a = 1.0 - a
    lo, hi = np.percentile(a, 1), a.max()
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    a = np.clip((a - 0.25) / 0.75, 0, 1)             # drop paper texture and shadows
    ys, xs = np.nonzero(a > 0.1)
    if len(ys) == 0:
        raise SystemExit(f"{path}: no digit found (the image looks blank)")
    crop = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = 20.0 / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    img = Image.fromarray((crop * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS)
    box = np.zeros((28, 28), np.float32)
    top, left = (28 - nh) // 2, (28 - nw) // 2
    box[top:top + nh, left:left + nw] = np.asarray(img, np.float32) / 255.0
    yy, xx = np.indices(box.shape)
    m = box.sum()
    dy, dx = round(14 - (yy * box).sum() / m), round(14 - (xx * box).sum() / m)
    out = np.zeros_like(box)
    ys0, ys1 = max(0, dy), min(28, 28 + dy)
    xs0, xs1 = max(0, dx), min(28, 28 + dx)
    out[ys0:ys1, xs0:xs1] = box[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return (out * 255).round().astype(np.uint8)


def ascii_art(x16: np.ndarray) -> str:
    ramp = " .:-=+*#%@"
    v = (x16 - x16.min()) / max(x16.max() - x16.min(), 1e-6)
    return "\n".join("    " + "".join(ramp[min(9, int(p * 10))] * 2 for p in row) for row in v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("--sim", action="store_true", help="also run it on the Atalla sim (about 75 minutes)")
    ap.add_argument("--kernel-objs", type=Path, help="with --sim: reuse compiled objects from an earlier build")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from train_resnet_small import prepare
    from model.resnet_small import trained

    d28 = to_mnist(args.image)
    x = prepare(d28[None])                          # (1, 3, 16, 16), exactly as in training
    stem = args.image.with_suffix("")
    Image.fromarray(d28).resize((280, 280), Image.NEAREST).save(f"{stem}_28x28.png")
    x16 = x[0, 0].numpy() * 0.3081 + 0.1307
    Image.fromarray((np.clip(x16, 0, 1) * 255).astype(np.uint8)).resize((256, 256), Image.NEAREST).save(f"{stem}_16x16.png")
    print(f"{args.image} as the model sees it (16x16):\n{ascii_art(x16)}\n")

    results = {}
    for name, sparse in (("dense", False), ("2:4", True)):
        m = trained(sparse)
        with torch.no_grad():
            p32 = torch.softmax(m(x), 1)[0]
            pbf = torch.softmax(m.bfloat16()(x.bfloat16()).float(), 1)[0]
        top = torch.topk(p32, 3)
        results[name] = int(pbf.argmax())
        alts = ", ".join(f"{int(i)} {float(v):.1%}" for i, v in zip(top.indices[1:], top.values[1:]))
        print(f"{name:>6} model: {int(top.indices[0])}  ({float(top.values[0]):.1%} sure; next: {alts})"
              f"   bf16: {results[name]}")
    agree = "agree" if results["dense"] == results["2:4"] else "DISAGREE"
    print(f"\nThe dense and 2:4 models {agree}. What the model saw: {stem}_28x28.png, {stem}_16x16.png")

    if args.sim:
        npy = Path(f"{stem}_input.pt").resolve()
        torch.save(x, npy)
        env = dict(os.environ, MNIST_IMAGE=str(npy))
        out_dir = ROOT / "models" / "out" / f"digit_{stem.name}"
        cmd = [sys.executable, str(ROOT / "tools" / "live_sparsity.py"), "resnet_small_mnist_2to4",
               "--out-dir", str(out_dir), "--csv", f"{stem}_array_cycles.csv"]
        if args.kernel_objs:
            cmd += ["--kernel-objs", str(args.kernel_objs)]
        print(f"\nRunning it on the Atalla sim (about 75 minutes):\n  {' '.join(cmd)}\n", flush=True)
        sys.exit(subprocess.call(cmd, cwd=ROOT / "models", env=env))


if __name__ == "__main__":
    main()
