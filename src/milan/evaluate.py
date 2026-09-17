from __future__ import annotations

import numpy as np


def _pair(y: np.ndarray, yhat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64).ravel()
    yhat = np.asarray(yhat, dtype=np.float64).ravel()
    if y.shape != yhat.shape:
        raise ValueError(f"shape mismatch: y {y.shape} vs yhat {yhat.shape}")
    if y.size == 0:
        raise ValueError("cannot score an empty forecast")
    return y, yhat


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = _pair(y, yhat)
    return float(np.abs(y - yhat).mean())


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = _pair(y, yhat)
    return float(np.sqrt(((y - yhat) ** 2).mean()))


def mape(y: np.ndarray, yhat: np.ndarray, threshold: float = 1e-3) -> tuple[float, int]:
    """MAPE over targets above `threshold`, plus the number of excluded points.

    Night-time traffic approaches zero, where a small absolute error becomes an
    unbounded percentage. Rather than silently adding an epsilon -- which invents
    a denominator and hides the problem -- we exclude those targets and report how
    many were excluded, so the reader can judge the metric's coverage.
    """
    y, yhat = _pair(y, yhat)
    keep = y >= threshold
    excluded = int((~keep).sum())
    if not keep.any():
        return float("nan"), excluded
    return float((np.abs(y[keep] - yhat[keep]) / y[keep]).mean() * 100.0), excluded


def smape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Symmetric MAPE in [0, 200]. Bounded, and defined where y == yhat == 0."""
    y, yhat = _pair(y, yhat)
    denom = np.abs(y) + np.abs(yhat)
    ratio = np.divide(np.abs(y - yhat), denom, out=np.zeros_like(denom), where=denom > 0)
    return float(ratio.mean() * 200.0)


def seasonal_naive_mae_insample(train_series: np.ndarray, m: int = 144) -> float:
    """In-sample seasonal-naive MAE: the MASE denominator.

    Computed on the TRAINING split only, so the denominator carries no test
    information and MASE values stay comparable across models for one area.
    """
    series = np.asarray(train_series, dtype=np.float64).ravel()
    if series.size <= m:
        raise ValueError(f"need more than m={m} training points, got {series.size}")
    denom = float(np.abs(series[m:] - series[:-m]).mean())
    if denom == 0.0:
        raise ValueError("seasonal-naive denominator is zero; series is perfectly periodic")
    return denom


def mase(y: np.ndarray, yhat: np.ndarray, denom: float) -> float:
    if denom <= 0:
        raise ValueError(f"MASE denominator must be positive, got {denom}")
    return mae(y, yhat) / denom


def skill_score(mase_value: float) -> float:
    """1 - MASE. Positive beats seasonal-naive; 0 ties it; negative loses to it."""
    return 1.0 - mase_value


def all_metrics(
    y: np.ndarray, yhat: np.ndarray, denom: float, threshold: float = 1e-3
) -> dict:
    mape_value, excluded = mape(y, yhat, threshold)
    mase_value = mase(y, yhat, denom)
    return {
        "mae": mae(y, yhat),
        "rmse": rmse(y, yhat),
        "mape": mape_value,
        "mape_excluded": excluded,
        "smape": smape(y, yhat),
        "mase": mase_value,
        "skill": skill_score(mase_value),
    }

