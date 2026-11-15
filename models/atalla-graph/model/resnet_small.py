"""Small-scale ResNet-50 for Atalla pipeline testing, and a 2:4-pruned twin.

ResNet-50's shape -- 3x3 stem, max pool, bottleneck blocks (1x1 reduce, 3x3,
1x1 expand by 4, projection shortcut where the shape changes), global average
pool, one classifier -- shrunk to two stages of one block each so it runs on
the cycle-level sim in minutes.

Two layers ResNet needs have no Atalla kernel, so they are folded away before
tracing, exactly, rather than approximated:

  BatchNorm (inference)  w' = w * s,  b' = beta - mean * s,
                         s  = gamma / sqrt(var + eps)        per out channel
  global average pool    fc(avgpool(x)) = fc'(flatten(x)),
                         W' = repeat_interleave(W, H*W, dim=1) / (H*W)

ResNetSmallRef is the textbook network (BN, avgpool) and is what the fold is
tested against; ResNetSmall is the folded network the graph builder traces.
Its ops are conv2d (+bias), relu, add, max_pool2d, flatten and linear --
all on the kernel list.

Channels round UP to multiples of 32, floor 32: the planner pads every tile
to 32x32 anyway, so a 3-channel layer costs a 32-channel tile, and 2:4 only
saves passes when a layer's K exceeds one tile. At the default scale the
widths are stem 32, stage 1 32->128, stage 2 64->256.

sparsity="2:4" prunes every conv except the stem (K = 27, one tile either
way) to 2:4 along K by magnitude, after the fold. K is ordered (C_in, kh,
kw) -- weight.reshape(O, -1) -- which is the order the conv packer
(memoryallocator: reshape(out, -1).T) and the im2col plan (k -> ic, ky, kx)
both use, so each group of four is four consecutive lanes of one K tile.
The classifier stays dense: its K is inflated by the avgpool fold, so its
zeros would be an artefact of the fold, not of training.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

EXPANSION = 4
INPUT_HW = 16


def _sc(c: int, scale: float) -> int:
    return max(32, math.ceil(c * scale / 32) * 32)


def prune_2to4_(weight: torch.Tensor) -> torch.Tensor:
    """Zero the two smallest of every four along K = weight.reshape(O, -1),
    in place. Returns the keep mask in weight's shape."""
    out_ch = weight.shape[0]
    flat = weight.detach().reshape(out_ch, -1)
    if flat.shape[1] % 4:
        raise ValueError(f"K={flat.shape[1]} is not a multiple of 4")
    groups = flat.reshape(out_ch, -1, 4)
    keep = torch.zeros_like(groups, dtype=torch.bool)
    keep.scatter_(2, groups.abs().topk(2, dim=2).indices, True)
    mask = keep.reshape(weight.shape)
    with torch.no_grad():
        weight.mul_(mask.to(weight.dtype))
    return mask


def is_2to4(weight: torch.Tensor) -> bool:
    flat = weight.detach().reshape(weight.shape[0], -1, 4)
    return bool(((flat != 0).sum(dim=2) <= 2).all())


# -- the textbook network ----------------------------------------------------

def _randomize_bn(bn: nn.BatchNorm2d) -> None:
    """Untrained BN is the identity, which would make the fold untestable."""
    with torch.no_grad():
        bn.weight.uniform_(0.5, 1.5)
        bn.bias.normal_(0.0, 0.1)
        bn.running_mean.normal_(0.0, 0.1)
        bn.running_var.uniform_(0.5, 1.5)


class _ConvBN(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        _randomize_bn(self.bn)

    def forward(self, x):
        return self.bn(self.conv(x))


class _BottleneckRef(nn.Module):
    """ResNet-50 v1.5 bottleneck: the stride sits on the 3x3."""

    def __init__(self, cin: int, mid: int, stride: int):
        super().__init__()
        cout = mid * EXPANSION
        self.reduce = _ConvBN(cin, mid, 1)
        self.spatial = _ConvBN(mid, mid, 3, stride)
        self.expand = _ConvBN(mid, cout, 1)
        self.shortcut = _ConvBN(cin, cout, 1, stride) if (stride != 1 or cin != cout) else None

    def forward(self, x):
        y = F.relu(self.reduce(x))
        y = F.relu(self.spatial(y))
        y = self.expand(y)
        s = x if self.shortcut is None else self.shortcut(x)
        return F.relu(y + s)


class ResNetSmallRef(nn.Module):
    def __init__(self, scale: float = 0.01, num_classes: int = 10):
        super().__init__()
        # ResNet-50 doubles the width at every stage (mid 64, 128, 256, 512).
        # Scale the base width, then double it, so rounding up to 32 cannot
        # flatten the stages into one width.
        stem = base = _sc(64, scale)
        mids = [base, 2 * base]
        self.stem = _ConvBN(3, stem, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.stage1 = _BottleneckRef(stem, mids[0], 1)
        self.stage2 = _BottleneckRef(mids[0] * EXPANSION, mids[1], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(mids[1] * EXPANSION, num_classes)
        self.final_hw = (INPUT_HW // 2 // 2) ** 2      # pool, then the stride-2 stage

    def forward(self, x):
        x = self.pool(F.relu(self.stem(x)))
        x = self.stage2(self.stage1(x))
        return self.fc(torch.flatten(self.avgpool(x), 1))


# -- the folded network the graph builder traces -----------------------------

def _fold(cbn: _ConvBN) -> nn.Conv2d:
    conv, bn = cbn.conv, cbn.bn
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    out = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                    stride=conv.stride, padding=conv.padding, bias=True)
    with torch.no_grad():
        out.weight.copy_(conv.weight * s[:, None, None, None])
        out.bias.copy_(bn.bias - bn.running_mean * s)
    return out


class _Bottleneck(nn.Module):
    def __init__(self, ref: _BottleneckRef):
        super().__init__()
        self.reduce = _fold(ref.reduce)
        self.spatial = _fold(ref.spatial)
        self.expand = _fold(ref.expand)
        self.shortcut = _fold(ref.shortcut) if ref.shortcut is not None else None

    def forward(self, x):
        y = F.relu(self.reduce(x))
        y = F.relu(self.spatial(y))
        y = self.expand(y)
        s = x if self.shortcut is None else self.shortcut(x)
        return F.relu(y + s)

    def convs(self):
        return [c for c in (self.reduce, self.spatial, self.expand, self.shortcut) if c is not None]


class ResNetSmall(nn.Module):
    """The folded network. Built from a freshly initialised ResNetSmallRef, so
    under a fixed torch seed it is deterministic."""

    def __init__(self, scale: float = 0.01, num_classes: int = 10,
                 sparsity: Optional[str] = None, ref: Optional[ResNetSmallRef] = None):
        super().__init__()
        ref = (ref or ResNetSmallRef(scale, num_classes)).eval()
        self.stem = _fold(ref.stem)
        self.pool = nn.MaxPool2d(2, 2)
        self.stage1 = _Bottleneck(ref.stage1)
        self.stage2 = _Bottleneck(ref.stage2)
        hw = ref.final_hw
        self.fc = nn.Linear(ref.fc.in_features * hw, num_classes)
        with torch.no_grad():
            # NCHW flatten puts channel c's pixels at c*HW .. c*HW+HW-1.
            self.fc.weight.copy_(ref.fc.weight.repeat_interleave(hw, dim=1) / hw)
            self.fc.bias.copy_(ref.fc.bias)
        if sparsity not in (None, "2:4"):
            raise ValueError(f"unknown sparsity {sparsity!r}")
        self.sparsity = sparsity
        if sparsity == "2:4":
            for conv in self.pruned_convs():
                prune_2to4_(conv.weight)

    def pruned_convs(self):
        """The convs 2:4 applies to: every one but the stem."""
        return self.stage1.convs() + self.stage2.convs()

    def forward(self, x):
        x = self.pool(F.relu(self.stem(x)))
        x = self.stage2(self.stage1(x))
        return self.fc(torch.flatten(x, 1))


def example_input() -> torch.Tensor:
    return torch.randn(1, 3, INPUT_HW, INPUT_HW)


# -- trained on MNIST (train_resnet_small.py) ----------------------------------

_WEIGHTS = __import__("pathlib").Path(__file__).resolve().parent / "weights"


def trained(sparse: bool = False) -> ResNetSmall:
    """resnet_small with the MNIST-trained weights: dense, or the 2:4 twin
    pruned and fine-tuned with its mask held (98.95% vs 98.98% test)."""
    model = ResNetSmall()
    name = "resnet_small_mnist_2to4.pt" if sparse else "resnet_small_mnist.pt"
    model.load_state_dict(torch.load(_WEIGHTS / name, weights_only=True))
    if sparse:
        model.sparsity = "2:4"
        assert all(is_2to4(c.weight) for c in model.pruned_convs())
    return model.eval()


def mnist_sample(digit: int = 7) -> torch.Tensor:
    """The first MNIST test image of `digit`, preprocessed: (1, 3, 16, 16)."""
    samples = torch.load(_WEIGHTS / "mnist_samples.pt", weights_only=True)
    return samples["x"][digit:digit + 1].clone()
