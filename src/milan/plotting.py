from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless: no display on this host
import matplotlib.pyplot as plt
import numpy as np

from milan.config import Config
from milan.data.loader import load_area

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


def plot_predictions(cfg: Config, area: int, model: str, forecast, path: Path) -> None:
    """Actual vs predicted over the full test week. One of the nine required plots."""
    test = cfg.split_cols("test")
    truth = load_area(cfg, area)[test]
    axis = time_axis(cfg, test.start, test.stop)

    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.plot(axis, truth, color=MODEL_COLORS["actual"], label="Actual", linewidth=1.0)
    ax.plot(axis, np.asarray(forecast), color=MODEL_COLORS.get(model, "#1f6fb4"),
            label=f"{model} (predicted)", linewidth=1.0, alpha=0.85)
    ax.set(xlabel="Date (Dec 16-22, 2013)", ylabel="Internet activity",
           title=f"{model} — one-step-ahead forecast, square {area}")
    ax.legend(loc="upper right", ncols=2)
    save(fig, path)


def plot_all_models_zoom(cfg: Config, area: int, forecasts: dict, day_offset: int,
                         path: Path) -> None:
    """All models over one 24-hour window.

    A 1008-point week-long plot hides exactly the phase behaviour the failure
    analysis is about, so every week plot is paired with one zoomed day.
    """
    test = cfg.split_cols("test")
    c0 = test.start + day_offset * cfg.slots_per_day
    c1 = c0 + cfg.slots_per_day
    truth = load_area(cfg, area)[c0:c1]
    axis = time_axis(cfg, c0, c1)
    offset = c0 - test.start

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(axis, truth, color=MODEL_COLORS["actual"], label="Actual", linewidth=1.6)
    for name, values in forecasts.items():
        ax.plot(axis, np.asarray(values)[offset: offset + cfg.slots_per_day],
                color=MODEL_COLORS.get(name), label=name, alpha=0.85)
    date = cfg.dates()[cfg.test_days[0] + day_offset]
    ax.set(xlabel=f"{date} (24 h)", ylabel="Internet activity",
           title=f"All models, square {area} — {date}")
    ax.legend(loc="upper left", ncols=3)
    save(fig, path)
