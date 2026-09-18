from __future__ import annotations

from torch import nn

from milan.models.dlinear import DLinear
from milan.models.gru import GRUForecaster
from milan.models.tcn import TCN

MODELS: dict[str, type[nn.Module]] = {
    "dlinear": DLinear,
    "tcn": TCN,
    "gru": GRUForecaster,
}


def build_model(name: str, L: int, **hyperparams) -> nn.Module:
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](L=L, **hyperparams)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


