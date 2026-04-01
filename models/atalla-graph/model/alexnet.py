
from __future__ import annotations

import torch.nn as nn


def build_alexnet() -> nn.Module:
    """Return an AlexNet instance without pretrained weights."""
    try:
        from torchvision.models import alexnet
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "torchvision is required to build AlexNet; install torchvision to enable "
            "the --model alexnet option."
        ) from exc

    model = alexnet(weights=None)
    model.eval()
    return model
