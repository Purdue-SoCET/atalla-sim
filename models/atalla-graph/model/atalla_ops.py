"""Graph leaf modules: FX traces these as a single call_module (see run_graph Tracer)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("AtallaSdpa",)


class AtallaSdpa(nn.Module):
    """Single-head attention (Q@K^T) * 1/√d → softmax → @V. Emitter tag atalla_sdpa; hardware when N=D=32."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.inv_sqrt_d = 1.0 / math.sqrt(float(dim))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        s = (q @ k.transpose(-2, -1)) * self.inv_sqrt_d
        p = F.softmax(s, dim=-1)
        return p @ v
