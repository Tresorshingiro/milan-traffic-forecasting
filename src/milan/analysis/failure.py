"""Diagnostics for where and why the forecasts fail.

The central question is whether a model has learned structure or merely learned to
repeat the previous value. One-step-ahead forecasting rewards persistence heavily
enough that a model can post a respectable MAE while contributing nothing, so the
phase-lag and residual-autocorrelation diagnostics below are not optional extras --
they are what separates a real result from an artefact.
"""
from __future__ import annotations

import numpy as np

from milan.analysis.profiling import acf_at
from milan.config import Config
from milan.evaluate import mae


def _pair(truth, forecast):
    truth = np.asarray(truth, dtype=np.float64).ravel()
    forecast = np.asarray(forecast, dtype=np.float64).ravel()
    if truth.shape != forecast.shape:
        raise ValueError(f"shape mismatch: {truth.shape} vs {forecast.shape}")
    return truth, forecast


def residual_autocorrelation(truth, forecast, lags=(1, 2, 144)) -> dict:
    """ACF of the residuals at selected lags.

    Well-specified one-step forecasts leave approximately white residuals. Strong
    residual autocorrelation at lag 1 means systematic timing error; strong
    autocorrelation at lag 144 means the model never learned the daily cycle and
    left it in the error.
    """
    truth, forecast = _pair(truth, forecast)
    residual = truth - forecast
    if np.allclose(residual, residual[0]):
        return {f"acf_{lag}": 0.0 for lag in lags}
    return {f"acf_{lag}": acf_at(residual, lag) for lag in lags}


def phase_lag_diagnostic(truth, forecast, max_shift: int = 3) -> dict:
    """Does shifting the forecast forward in time reduce its error?

    If MAE is minimised at a non-zero shift, the forecast is a delayed copy of
    reality -- the signature of a model that has settled on persistence. Shifts are
    compared on the same overlapping window so the numbers are commensurable.
    """
    truth, forecast = _pair(truth, forecast)
    if max_shift < 1 or truth.size <= max_shift:
        raise ValueError("max_shift must be >= 1 and shorter than the series")

    # A lagging forecast satisfies forecast[t] ~ truth[t-1]. To detect that we must
    # compare truth[i] against forecast[i + shift] -- pulling the forecast EARLIER in
    # index. Shifting the other way makes a lagging forecast look worse at every shift
    # and reports best_shift = 0, a false negative on the very failure this detects.
    # Every shift is scored on a window of identical length, so the MAEs are comparable.
    mae_by_shift = {}
    span = truth.size - max_shift
    for shift in range(0, max_shift + 1):
        aligned_truth = truth[:span]
        aligned_forecast = forecast[shift: span + shift]
        mae_by_shift[shift] = mae(aligned_truth, aligned_forecast)

    best = min(mae_by_shift, key=mae_by_shift.get)
    return {
        "best_shift": int(best),
        "mae_by_shift": mae_by_shift,
        "interpretation": (
            "forecast is a delayed copy of reality (persistence-like)"
            if best > 0 else
            "no systematic timing offset detected"
        ),
    }


def error_by_slot(cfg: Config, truth, forecast) -> np.ndarray:
    """Mean absolute error for each of the 144 daily slots, averaged over test days."""
    truth, forecast = _pair(truth, forecast)
    errors = np.abs(truth - forecast)
    n_days = errors.size // cfg.slots_per_day
    return errors[: n_days * cfg.slots_per_day].reshape(n_days, cfg.slots_per_day).mean(axis=0)


def error_by_day(cfg: Config, truth, forecast) -> np.ndarray:
    """Mean absolute error for each test day. Exposes end-of-week regime drift."""
    truth, forecast = _pair(truth, forecast)
    errors = np.abs(truth - forecast)
    n_days = errors.size // cfg.slots_per_day
    return errors[: n_days * cfg.slots_per_day].reshape(n_days, cfg.slots_per_day).mean(axis=1)


def worst_windows(cfg: Config, truth, forecast, n: int = 10) -> list[dict]:
    """The n single steps with the largest absolute error, with their timestamps."""
    truth, forecast = _pair(truth, forecast)
    errors = np.abs(truth - forecast)
    test = cfg.split_cols("test")
    start = np.datetime64(f"{cfg.start_date.isoformat()}T00:00")

    out = []
    for index in np.argsort(errors)[::-1][:n]:
        absolute_column = test.start + int(index)
        stamp = start + absolute_column * np.timedelta64(10, "m")
        out.append({
            "index": int(index),
            "timestamp": str(stamp),
            "actual": float(truth[index]),
            "predicted": float(forecast[index]),
            "abs_error": float(errors[index]),
        })
    return out
