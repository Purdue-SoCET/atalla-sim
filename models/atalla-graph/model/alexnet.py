"""Small-scale AlexNet for Atalla pipeline testing.

Uses scaled-down channel counts to fit within emulator constraints.
FX-traceable (no control flow, no dynamic shapes).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNetSmall(nn.Module):
    """AlexNet with configurable channel scaling and smaller input (32x32)."""

    def __init__(self, scale: float = 0.01, num_classes: int = 10):
        super().__init__()
        def sc(c: int) -> int:
            return max(1, int(c * scale))

        self.conv1 = nn.Conv2d(3, sc(64), kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(sc(64), sc(192), kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(sc(192), sc(384), kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(sc(384), sc(256), kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(sc(256), sc(256), kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        fc_in = sc(256) * 4 * 4  # after 3 pool layers: 32->16->8->4
        self.fc1 = nn.Linear(fc_in, sc(4096))
        self.fc2 = nn.Linear(sc(4096), sc(4096))
        self.fc3 = nn.Linear(sc(4096), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
