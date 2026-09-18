from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class Chomp(nn.Module):
    """Trim right-hand padding so each convolution stays causal."""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.size] if self.size > 0 else x


class TCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation)),
            Chomp(pad), nn.ReLU(), nn.Dropout(dropout),
            weight_norm(nn.Conv1d(c_out, c_out, kernel, padding=pad, dilation=dilation)),
            Chomp(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        return self.activation(self.net(x) + residual)


class TCN(nn.Module):
    def __init__(
        self,
        L: int,
        channels: int = 32,
        kernel: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.receptive_field = 1 + 2 * (kernel - 1) * sum(dilations)
        if self.receptive_field < L:
            raise ValueError(
                f"receptive field {self.receptive_field} < input length {L}: the model "
                f"could not see its whole window. Add dilations or increase the kernel."
            )
        blocks, c_in = [], 1
        for dilation in dilations:
            blocks.append(TCNBlock(c_in, channels, kernel, dilation, dropout))
            c_in = channels
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, L) -> (B,)
        features = self.blocks(x.unsqueeze(1))            # (B, C, L)
        return self.head(features[:, :, -1]).squeeze(-1)  # last step only

    