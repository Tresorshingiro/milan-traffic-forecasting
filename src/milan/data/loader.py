from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from milan.config import Config


def open_matrix(cfg: Config) -> np.memmap:
    """Read-only memmap of the (10000, 8928) float32 traffic matrix."""
    if not cfg.matrix_path.exists():
        raise FileNotFoundError(
            f"{cfg.matrix_path} not found -- run `python scripts/build_matrix.py` first"
        )
    return np.load(cfg.matrix_path, mmap_mode="r")


def load_area(cfg: Config, square_id: int) -> np.ndarray:
    """One area's full series as float64. Touches ~36 KB of the 357 MB matrix."""
    if not 1 <= square_id <= cfg.n_squares:
        raise ValueError(f"square_id {square_id} outside 1..{cfg.n_squares}")
    return np.asarray(open_matrix(cfg)[square_id - 1], dtype=np.float64)


def load_square_totals(cfg: Config) -> np.ndarray:
    if not cfg.totals_path.exists():
        raise FileNotFoundError(
            f"{cfg.totals_path} not found -- run `python scripts/build_matrix.py` first"
        )
    return np.load(cfg.totals_path)


def top_areas(cfg: Config, n: int | None = None) -> list[int]:
    """1-based square ids with the highest total traffic, descending."""
    totals = load_square_totals(cfg)
    n = cfg.n_top if n is None else n
    return [int(i) + 1 for i in np.argsort(totals)[::-1][:n]]


def target_areas(cfg: Config) -> list[int]:
    busiest = top_areas(cfg, 1)[0]
    areas = [busiest] + [a for a in cfg.extra_areas if a != busiest]
    return areas[:3]


@dataclass(frozen=True)
class Scaler:
    """Affine scaler, optionally preceded by log1p. Fitted on training data only."""

    mu: float
    sigma: float
    use_log1p: bool

    def transform(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        if self.use_log1p:
            values = np.log1p(values)
        return (values - self.mu) / self.sigma

    def inverse(self, z: np.ndarray) -> np.ndarray:
        values = np.asarray(z, dtype=np.float64) * self.sigma + self.mu
        if self.use_log1p:
            values = np.expm1(values)
        return np.maximum(values, 0.0)


def fit_scaler(train_values: np.ndarray, use_log1p: bool) -> Scaler:
    values = np.asarray(train_values, dtype=np.float64)
    if use_log1p:
        values = np.log1p(values)
    sigma = float(values.std())
    if sigma == 0.0:
        raise ValueError("training split has zero variance; cannot fit a scaler")
    return Scaler(mu=float(values.mean()), sigma=sigma, use_log1p=use_log1p)


def make_windows(
    series: np.ndarray, L: int, start: int, end: int
) -> tuple[np.ndarray, np.ndarray]:
    """Windows predicting `series[t]` from `series[t-L:t]`, for t in [start, end).

    Returns X of shape (end-start, L) and y of shape (end-start,), both float32.
    """
    if L < 1:
        raise ValueError(f"L must be >= 1, got {L}")
    if start - L < 0:
        raise ValueError(
            f"insufficient history: need {L} steps before index {start}, "
            f"but the series starts at 0"
        )
    if end > len(series):
        raise ValueError(f"end {end} exceeds series length {len(series)}")

    windows = np.lib.stride_tricks.sliding_window_view(series, L)
    X = windows[start - L: end - L]
    y = series[start:end]
    return (np.ascontiguousarray(X, dtype=np.float32),
            np.ascontiguousarray(y, dtype=np.float32))