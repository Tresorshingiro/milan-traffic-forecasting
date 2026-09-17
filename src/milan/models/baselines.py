from __future__ import annotations

import numpy as np


def _check(series: np.ndarray, start: int, end: int, lag: int) -> np.ndarray:
    values = np.asarray(series, dtype=np.float64).ravel()
    if start - lag < 0:
        raise ValueError(
            f"insufficient history: need {lag} steps before index {start}"
        )
    if end > values.size:
        raise ValueError(f"end {end} exceeds series length {values.size}")
    return values


def persistence(series: np.ndarray, start: int, end: int) -> np.ndarray:
    """x_hat(t) = x(t-1). The lower bound on competence."""
    values = _check(series, start, end, 1)
    return values[start - 1: end - 1]


def seasonal_naive(series: np.ndarray, start: int, end: int, m: int = 144) -> np.ndarray:
    """x_hat(t) = x(t-m). With m=144 this is the same 10-minute slot yesterday."""
    values = _check(series, start, end, m)
    return values[start - m: end - m]


