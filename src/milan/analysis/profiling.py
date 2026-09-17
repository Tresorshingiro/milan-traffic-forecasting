from __future__ import annotations

import math

import numpy as np


def _clean(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64).ravel()
    if values.size < 100:
        raise ValueError(f"series too short to characterise: {values.size} points")
    if not np.isfinite(values).all():
        raise ValueError("series contains non-finite values")
    return values


# --- dispersion ----------------------------------------------------------

def coefficient_of_variation(x: np.ndarray) -> float:
    values = _clean(x)
    mean = values.mean()
    if mean == 0:
        raise ValueError("cannot compute CV of a zero-mean series")
    return float(values.std() / mean)


def fano_factor(x: np.ndarray) -> float:
    """Variance-to-mean ratio. 1.0 for a Poisson process; > 1 is over-dispersed."""
    values = _clean(x)
    mean = values.mean()
    if mean == 0:
        raise ValueError("cannot compute Fano factor of a zero-mean series")
    return float(values.var() / mean)


def burstiness(x: np.ndarray) -> float:
    """Goh & Barabasi (2008) burstiness in [-1, 1].

    -1 is perfectly regular, 0 matches a Poisson process, and positive values
    indicate bursty, event-driven activity.
    """
    values = _clean(x)
    std, mean = values.std(), values.mean()
    if std + mean == 0:
        raise ValueError("cannot compute burstiness of an all-zero series")
    return float((std - mean) / (std + mean))


def gini(x: np.ndarray) -> float:
    """Gini coefficient in [0, 1). Used for spatial traffic concentration."""
    values = np.sort(np.asarray(x, dtype=np.float64).ravel())
    if values.size == 0 or values.sum() == 0:
        raise ValueError("cannot compute Gini of an empty or all-zero vector")
    if (values < 0).any():
        raise ValueError("Gini is undefined for negative values")
    n = values.size
    cumulative = np.cumsum(values)
    return float((n + 1 - 2 * (cumulative / cumulative[-1]).sum()) / n)


# --- complexity ----------------------------------------------------------

def permutation_entropy(x: np.ndarray, m: int = 4, tau: int = 1) -> float:
    """Bandt & Pompe (2002) permutation entropy, normalised to [0, 1].

    Counts how many of the m! possible orderings of m consecutive (tau-spaced)
    samples actually occur, and how evenly. Near 0 means highly predictable
    ordinal structure; near 1 means the ordering is essentially random. It is
    robust to monotone transformations and to outliers, which matters for a
    bursty series where a handful of spikes dominate the variance.
    """
    values = _clean(x)
    if m < 2:
        raise ValueError(f"m must be >= 2, got {m}")
    width = (m - 1) * tau + 1
    if values.size < width + 1:
        raise ValueError(f"series shorter than the embedding width {width}")
    windows = np.lib.stride_tricks.sliding_window_view(values, width)[:, ::tau]
    patterns = np.argsort(windows, axis=1, kind="stable")
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(math.factorial(m)))


def spectral_entropy(x: np.ndarray) -> float:
    """Normalised Shannon entropy of the power spectrum, in [0, 1].

    0 means all power sits in one frequency (a pure tone); 1 means power is spread
    evenly (white noise). The DC component is removed so the measure describes
    variability, not level.
    """
    values = _clean(x)
    power = np.abs(np.fft.rfft(values - values.mean())) ** 2
    power = power[1:]                       # drop DC
    total = power.sum()
    if total == 0:
        raise ValueError("series has no spectral power (constant series)")
    probabilities = power / total
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(probabilities.size))


def dfa_exponent(
    x: np.ndarray, min_scale: int = 10, max_scale: int | None = None, n_scales: int = 20
) -> float:
    """Detrended fluctuation analysis exponent alpha (Peng et al., 1994).

    The input is integrated into a profile, then the RMS of linear-detrended
    fluctuations is regressed against window size in log-log space.

    Convention: alpha ~ 0.5 for white noise, alpha ~ 1.5 for an already-integrated
    signal (alpha = H + 1 in that case). For a noise-like series such as traffic,
    alpha estimates the Hurst exponent directly; 0.5 < alpha < 1 indicates
    persistent long-range correlations.
    """
    values = _clean(x)
    n = values.size
    max_scale = n // 4 if max_scale is None else max_scale
    if max_scale <= min_scale:
        raise ValueError(f"max_scale {max_scale} must exceed min_scale {min_scale}")

    profile = np.cumsum(values - values.mean())
    scales = np.unique(
        np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales).astype(int)
    )
    fluctuations = []
    for scale in scales:
        n_segments = n // scale
        segments = profile[: n_segments * scale].reshape(n_segments, scale).T
        time = np.arange(scale)
        design = np.vstack([time, np.ones(scale)]).T
        coefficients, *_ = np.linalg.lstsq(design, segments, rcond=None)
        residuals = segments - design @ coefficients
        fluctuations.append(float(np.sqrt((residuals ** 2).mean())))

    return float(np.polyfit(np.log(scales), np.log(fluctuations), 1)[0])


# --- seasonality ---------------------------------------------------------

def acf_at(x: np.ndarray, lag: int) -> float:
    """Sample autocorrelation at a single lag."""
    values = _clean(x)
    if not 0 < lag < values.size:
        raise ValueError(f"lag {lag} outside 1..{values.size - 1}")
    centred = values - values.mean()
    denominator = float((centred ** 2).sum())
    if denominator == 0:
        raise ValueError("cannot compute ACF of a constant series")
    return float((centred[lag:] * centred[:-lag]).sum() / denominator)


def acf_ratio(x: np.ndarray, m: int = 144) -> float:
    """acf(m) / acf(1): how much of the dependence is seasonal rather than local.

    A value near 1 means yesterday's value at this slot is as informative as the
    immediately preceding value -- the signature of a strongly periodic area.
    """
    local = acf_at(x, 1)
    if local == 0:
        raise ValueError("lag-1 autocorrelation is zero; ratio undefined")
    return float(acf_at(x, m) / local)


def seasonal_strength(x: np.ndarray, period: int = 144) -> float:
    """STL seasonal strength in [0, 1]: 1 - Var(remainder) / Var(seasonal + remainder).

    Follows Wang et al. / Hyndman's feature definition. `robust=False` is used for
    speed: on 8928 points the robust variant's iterations cost minutes for a value
    that moves in the third decimal place.
    """
    from statsmodels.tsa.seasonal import STL

    values = _clean(x)
    if values.size < 2 * period:
        raise ValueError(f"need at least {2 * period} points for period {period}")
    result = STL(values, period=period, robust=False).fit()
    seasonal_plus_remainder = result.seasonal + result.resid
    denominator = float(np.var(seasonal_plus_remainder))
    if denominator == 0:
        return 0.0
    return float(max(0.0, 1.0 - np.var(result.resid) / denominator))


def profile_area(x: np.ndarray, period: int = 144) -> dict:
    """Full forecastability profile for one area's series."""
    values = _clean(x)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "cv": coefficient_of_variation(values),
        "fano": fano_factor(values),
        "burstiness": burstiness(values),
        "seasonal_strength": seasonal_strength(values, period),
        "perm_entropy": permutation_entropy(values),
        "spectral_entropy": spectral_entropy(values),
        "dfa_alpha": dfa_exponent(values),
        "acf_1": acf_at(values, 1),
        "acf_144": acf_at(values, period),
        "acf_ratio": acf_ratio(values, period),
    }

