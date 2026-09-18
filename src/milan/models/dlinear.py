from __future__ import annotations

import torch
from torch import nn


class MovingAvg(nn.Module):

    def __init__(self, kernel: int) -> None:
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError(f"kernel must be odd to centre the average, got {kernel}")
        self.kernel = kernel
        self.pool = nn.AvgPool1d(kernel, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, L) -> (B, L)
        pad = (self.kernel - 1) // 2
        front = x[:, :1].repeat(1, pad)
        back = x[:, -1:].repeat(1, pad)
        padded = torch.cat([front, x, back], dim=1).unsqueeze(1)
        return self.pool(padded).squeeze(1)


class DLinear(nn.Module):
    def __init__(self, L: int, kernel: int = 25) -> None:
        super().__init__()
        self.moving_avg = MovingAvg(kernel)
        self.trend = nn.Linear(L, 1)
        self.seasonal = nn.Linear(L, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, L) -> (B,)
        trend = self.moving_avg(x)
        seasonal = x - trend
        return (self.trend(trend) + self.seasonal(seasonal)).squeeze(-1)

    