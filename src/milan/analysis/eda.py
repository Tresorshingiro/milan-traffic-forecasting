from __future__ import annotations


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from milan.analysis.profiling import acf_at, gini
from milan.config import Config
from milan.data.loader import load_area
from milan.plotting import save, time_axis


def fig_traffic_distribution(totals: np.ndarray, path: Path) -> dict:
    """Task 2 item 1: distribution of total internet traffic across all areas."""
    values = np.asarray(totals, dtype=np.float64)
    positive = values[values > 0]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].hist(values, bins=80, color="#1f6fb4", alpha=0.85)
    axes[0].set(xlabel="Total internet activity (62 days)", ylabel="Number of areas",
                title="Linear scale")
    axes[1].hist(np.log10(positive), bins=80, color="#2e8b57", alpha=0.85)
    axes[1].set(xlabel="log10(total internet activity)", ylabel="Number of areas",
                title="Log scale")
    fig.suptitle("Spatial distribution of total Internet traffic across 10,000 Milan areas")
    save(fig, path)

    order = np.sort(values)[::-1]
    top_1pct = int(0.01 * values.size)
    mean, std = values.mean(), values.std()
    return {
        "n_areas": int(values.size),
        "n_zero_areas": int((values == 0).sum()),
        "mean": float(mean),
        "median": float(np.median(values)),
        "std": float(std),
        "skew": float(((values - mean) ** 3).mean() / std ** 3),
        "gini": gini(values),
        "top_1pct_share": float(order[:top_1pct].sum() / values.sum()),
        "max_over_median": float(values.max() / np.median(values)),
    }


def fig_lorenz(totals: np.ndarray, path: Path) -> float:
    """Lorenz curve with Gini: turns 'traffic is concentrated' into a measurement."""
    values = np.sort(np.asarray(totals, dtype=np.float64))
    share_of_areas = np.arange(1, values.size + 1) / values.size
    share_of_traffic = np.cumsum(values) / values.sum()
    coefficient = gini(values)

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot([0, 1], [0, 1], "--", color="#999999", label="Perfect equality")
    ax.plot(share_of_areas, share_of_traffic, color="#1f6fb4",
            label=f"Milan areas (Gini = {coefficient:.3f})")
    ax.fill_between(share_of_areas, share_of_traffic, share_of_areas,
                    color="#1f6fb4", alpha=0.12)
    ax.set(xlabel="Cumulative share of areas", ylabel="Cumulative share of traffic",
           title="Concentration of Internet traffic")
    ax.legend(loc="upper left")
    save(fig, path)
    return coefficient


def fig_area_series(cfg: Config, areas: list[int], day0: int, day1: int, path: Path) -> None:
    """Task 2 items 2-3: series for the five required areas over the first two weeks."""
    columns = cfg.day_cols(day0, day1)
    axis = time_axis(cfg, columns.start, columns.stop)

    fig, axes = plt.subplots(len(areas), 1, figsize=(10, 1.7 * len(areas)), sharex=True)
    for ax, area in zip(np.atleast_1d(axes), areas):
        ax.plot(axis, load_area(cfg, area)[columns], color="#1f6fb4")
        ax.set_ylabel(f"Sq {area}", rotation=0, ha="right", va="center")
    np.atleast_1d(axes)[-1].set_xlabel("Date")
    fig.suptitle(f"Internet traffic, {cfg.dates()[day0]} to {cfg.dates()[day1 - 1]}")
    save(fig, path)


def fig_stl(series: np.ndarray, period: int, path: Path) -> float:
    """STL decomposition. Returns seasonal strength."""
    from statsmodels.tsa.seasonal import STL

    result = STL(np.asarray(series, dtype=np.float64), period=period, robust=False).fit()
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    for ax, (values, label) in zip(axes, [
        (result.observed, "Observed"), (result.trend, "Trend"),
        (result.seasonal, "Seasonal"), (result.resid, "Remainder"),
    ]):
        ax.plot(values, color="#1f6fb4", linewidth=0.7)
        ax.set_ylabel(label)
    axes[-1].set_xlabel("Time step (10-minute slots)")
    fig.suptitle(f"STL decomposition, period = {period} slots (24 h)")
    save(fig, path)

    # Seasonal strength from the decomposition already fitted above, rather than
    # calling profiling.seasonal_strength and paying for a second STL fit.
    seasonal_plus_remainder = result.seasonal + result.resid
    denominator = float(np.var(seasonal_plus_remainder))
    if denominator == 0:
        return 0.0
    return float(max(0.0, 1.0 - np.var(result.resid) / denominator))


def fig_acf_pacf(series: np.ndarray, nlags: int, path: Path) -> dict:
    """ACF and PACF. This figure is the evidence that sets the model input length."""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    values = np.asarray(series, dtype=np.float64)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    plot_acf(values, lags=nlags, ax=axes[0], zero=False)
    axes[0].set_title(f"Autocorrelation up to {nlags} lags ({nlags / 144:.0f} days)")
    plot_pacf(values, lags=min(nlags, 200), ax=axes[1], zero=False, method="ywm")
    axes[1].set_title("Partial autocorrelation")
    for ax in axes:
        for day in range(1, nlags // 144 + 1):
            ax.axvline(day * 144, color="#c77d1a", linestyle="--", alpha=0.6)
    save(fig, path)

    correlations = np.array([acf_at(values, lag) for lag in range(1, min(nlags, 500))])
    crossings = np.where(correlations <= 0)[0]
    return {
        "acf_1": acf_at(values, 1),
        "acf_144": acf_at(values, 144),
        "acf_288": acf_at(values, 288),
        "first_zero_crossing": int(crossings[0] + 1) if crossings.size else None,
    }


def stationarity_tests(series: np.ndarray) -> dict:
    """ADF and KPSS together.

    The two tests have opposite null hypotheses -- ADF's null is a unit root
    (non-stationarity), KPSS's null is stationarity -- so running both and reading
    the combination is far more informative than a single ADF call. The four
    possible outcomes are distinguished explicitly below, including the two
    inconclusive ones that a single test would silently hide.
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    values = np.asarray(series, dtype=np.float64)
    adf_stat, adf_p, *_ = adfuller(values, autolag="AIC")
    kpss_stat, kpss_p, *_ = kpss(values, regression="c", nlags="auto")

    adf_rejects = adf_p < 0.05          # rejects unit root => looks stationary
    kpss_rejects = kpss_p < 0.05        # rejects stationarity => looks non-stationary

    if adf_rejects and not kpss_rejects:
        interpretation = ("Both tests agree the series is stationary "
                          "(ADF rejects a unit root; KPSS does not reject stationarity).")
    elif not adf_rejects and kpss_rejects:
        interpretation = ("Both tests agree the series is non-stationary "
                          "(ADF fails to reject a unit root; KPSS rejects stationarity).")
    elif adf_rejects and kpss_rejects:
        interpretation = ("Inconclusive: both reject. This is the signature of a "
                          "trend-stationary series -- stationary about a deterministic "
                          "trend rather than around a constant mean.")
    else:
        interpretation = ("Inconclusive: neither rejects. The sample is not informative "
                          "enough to separate the hypotheses, often a sign of long memory.")

    return {
        "adf_stat": float(adf_stat), "adf_p": float(adf_p),
        "kpss_stat": float(kpss_stat), "kpss_p": float(kpss_p),
        "interpretation": interpretation,
    }


def fig_periodogram(series: np.ndarray, path: Path, top_n: int = 5) -> list[float]:
    """Power spectrum. Returns the dominant periods in hours."""
    values = np.asarray(series, dtype=np.float64)
    power = np.abs(np.fft.rfft(values - values.mean())) ** 2
    freqs = np.fft.rfftfreq(values.size, d=1.0)        # cycles per 10-minute slot
    power[0] = 0.0

    with np.errstate(divide="ignore"):
        periods_hours = 1.0 / freqs / 6.0              # 6 slots per hour

    fig, ax = plt.subplots(figsize=(7, 3.2))
    valid = (periods_hours > 0.5) & (periods_hours < 200)
    ax.semilogx(periods_hours[valid], power[valid], color="#1f6fb4")
    for marker, label in ((24, "24 h"), (12, "12 h"), (168, "1 week")):
        ax.axvline(marker, color="#c77d1a", linestyle="--", alpha=0.7)
        ax.text(marker, ax.get_ylim()[1] * 0.9, label, fontsize=8, ha="center")
    ax.set(xlabel="Period (hours, log scale)", ylabel="Spectral power",
           title="Periodogram")
    save(fig, path)

    dominant = np.argsort(power)[::-1][:top_n]
    return [float(periods_hours[i]) for i in dominant if np.isfinite(periods_hours[i])]



