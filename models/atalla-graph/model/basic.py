import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicModule(nn.Module):

    def __init__(self, dim: int = 64, depth: int = 4):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.randn(dim, dim) * 0.1)
                for _ in range(depth)
            ]
        )
        self.biases = nn.ParameterList(
            [
                nn.Parameter(torch.randn(dim) * 0.1)
                for _ in range(depth)
            ]
        )
        self.output = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            residual = out
            out = torch.matmul(out, weight)
            out = out + bias
            out = F.relu(out)
            if idx % 2 == 1:
                out = out + 0.5 * residual
        out = self.output(out)
        return out
