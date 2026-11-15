#!/usr/bin/env python3
"""Train resnet_small on MNIST, then make its 2:4 twin the standard way.

    python train_resnet_small.py --data DIR [--epochs 3] [--finetune 2]

DIR holds the four MNIST idx .gz files. Digits are resized 28x28 -> 16x16
and the grey channel is copied to all three, so the network -- and the ELF
built from it -- is exactly resnet_small.

  1. train ResNetSmallRef (real BatchNorm, real global average pool);
  2. fold it to the Atalla form and measure it;
  3. 2:4 one-shot: prune the folded convs (all but the stem) and measure,
     with no retraining -- the number that shows why recipes fine-tune;
  4. 2:4 fine-tuned (ASP-style): prune the reference network's convs, train
     on with the zeros held at zero, fold, measure.

Accuracy is measured in fp32 and in bf16 (what Atalla computes in). The
folded dense and fine-tuned 2:4 weights are saved to model/weights/.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model.resnet_small import (ResNetSmall, ResNetSmallRef, INPUT_HW,
                                is_2to4, prune_2to4_)

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "model" / "weights"


def _idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def prepare(images: np.ndarray) -> torch.Tensor:
    """uint8 (N,28,28) -> float (N,3,16,16), normalised like the usual MNIST recipe."""
    x = torch.from_numpy(images.astype(np.float32) / 255.0).unsqueeze(1)
    x = F.interpolate(x, size=(INPUT_HW, INPUT_HW), mode="bilinear", align_corners=False)
    x = (x - 0.1307) / 0.3081
    return x.repeat(1, 3, 1, 1)


def load(data: Path):
    xtr = prepare(_idx(data / "train-images-idx3-ubyte.gz"))
    ytr = torch.from_numpy(_idx(data / "train-labels-idx1-ubyte.gz").astype(np.int64))
    xte = prepare(_idx(data / "t10k-images-idx3-ubyte.gz"))
    yte = torch.from_numpy(_idx(data / "t10k-labels-idx1-ubyte.gz").astype(np.int64))
    return xtr, ytr, xte, yte


@torch.no_grad()
def accuracy(model, x, y, bf16=False, batch=1000) -> float:
    model = model.eval()
    if bf16:
        import copy
        model = copy.deepcopy(model).bfloat16()
    hits = 0
    for i in range(0, len(x), batch):
        xb = x[i:i + batch].bfloat16() if bf16 else x[i:i + batch]
        hits += int((model(xb).float().argmax(1) == y[i:i + batch]).sum())
    return hits / len(x)


def train(model, x, y, epochs, lr, masks=None, batch=128, tag=""):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps = epochs * ((len(x) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(x))
        t0, loss_sum = time.time(), 0.0
        for i in range(0, len(x), batch):
            idx = perm[i:i + batch]
            loss = F.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if masks:
                with torch.no_grad():
                    for w, m in masks:
                        w.mul_(m)
            loss_sum += float(loss) * len(idx)
        print(f"  {tag} epoch {ep + 1}/{epochs}: loss {loss_sum / len(x):.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)


def ref_pruned_convs(ref: ResNetSmallRef):
    return [cbn.conv for blk in (ref.stage1, ref.stage2)
            for cbn in (blk.reduce, blk.spatial, blk.expand, blk.shortcut) if cbn is not None]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--finetune", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    xtr, ytr, xte, yte = load(args.data)
    print(f"MNIST: {len(xtr)} train, {len(xte)} test, input {tuple(xtr.shape[1:])}", flush=True)
    results = {}

    def report(name, model):
        fp32, bf16 = accuracy(model, xte, yte), accuracy(model, xte, yte, bf16=True)
        results[name] = dict(fp32=fp32, bf16=bf16)
        print(f"{name:<28} fp32 {fp32:.2%}   bf16 {bf16:.2%}", flush=True)

    # 1-2. dense
    ref = ResNetSmallRef()
    train(ref, xtr, ytr, args.epochs, 2e-3, tag="dense")
    report("reference (BN, avgpool)", ref)
    dense = ResNetSmall(ref=ref)
    report("dense, folded", dense)

    # 3. one-shot 2:4 on the folded network
    oneshot = ResNetSmall(ref=ref, sparsity="2:4")
    report("2:4 one-shot", oneshot)

    # 4. ASP-style: prune the reference, fine-tune with the mask held, fold
    masks = [(conv.weight, prune_2to4_(conv.weight)) for conv in ref_pruned_convs(ref)]
    train(ref, xtr, ytr, args.finetune, 5e-4, masks=masks, tag="2:4 fine-tune")
    sparse = ResNetSmall(ref=ref)            # already 2:4; folding keeps the zeros
    assert all(is_2to4(c.weight) for c in sparse.pruned_convs())
    report("2:4 fine-tuned", sparse)

    WEIGHTS.mkdir(parents=True, exist_ok=True)
    torch.save(dense.state_dict(), WEIGHTS / "resnet_small_mnist.pt")
    torch.save(sparse.state_dict(), WEIGHTS / "resnet_small_mnist_2to4.pt")
    (WEIGHTS / "resnet_small_mnist.json").write_text(json.dumps(
        dict(seed=args.seed, epochs=args.epochs, finetune=args.finetune, results=results), indent=2) + "\n")
    print(f"saved weights to {WEIGHTS}")


if __name__ == "__main__":
    main()
