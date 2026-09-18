from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless: no display on this host
import matplotlib.pyplot as plt
import numpy as np

from milan.config import Config

# One ordered palette used everywhere, so a model keeps its colour across figures.
MODEL_COLORS = {
    "actual": "#222222",
    "persistence": "#9e9e9e",
    "seasonal_naive": "#c77d1a",
    "dlinear": "#1f6fb4",
    "tcn": "#2e8b57",
    "gru": "#b4326f",
}


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.2,
    })


def save(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def time_axis(cfg: Config, c0: int, c1: int) -> np.ndarray:
    """Datetime64 axis for columns [c0, c1), for readable x-axes."""
    start = np.datetime64(f"{cfg.start_date.isoformat()}T00:00")
    return start + np.arange(c0, c1) * np.timedelta64(10, "m")


