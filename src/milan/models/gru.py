from __future__ import annotations

import torch
from torch import nn


class GRUForecaster(nn.Module):
    def __init__(self, L: int, hidden: int = 64, layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.L = L
        self.gru = nn.GRU(
            input_size=1, hidden_size=hidden, num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,   # PyTorch ignores it for 1 layer
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, L) -> (B,)
        output, _ = self.gru(x.unsqueeze(-1))             # (B, L, H)
        return self.head(output[:, -1]).squeeze(-1)       # final hidden state


