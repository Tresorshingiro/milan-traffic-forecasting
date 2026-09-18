import numpy as np
import pytest

from milan.analysis.eda import (
    fig_acf_pacf, fig_lorenz, fig_periodogram, fig_traffic_distribution,
    stationarity_tests,
)

RNG = np.random.default_rng(0)


def test_lorenz_returns_gini_and_writes_a_figure(tmp_path):
    totals = RNG.lognormal(mean=3.0, sigma=1.5, size=10000)
    path = tmp_path / "lorenz.png"
    value = fig_lorenz(totals, path)
    assert 0.0 < value < 1.0
    assert path.exists() and path.stat().st_size > 1000


def test_distribution_summary_reports_heavy_tail_statistics(tmp_path):
    totals = RNG.lognormal(mean=3.0, sigma=1.5, size=10000)
    stats = fig_traffic_distribution(totals, tmp_path / "dist.png")
    assert set(stats) >= {"mean", "median", "std", "skew", "gini",
                          "top_1pct_share", "max_over_median"}
    # A log-normal is right-skewed, so the mean exceeds the median.
    assert stats["mean"] > stats["median"]
    assert stats["skew"] > 0
    assert 0.0 < stats["top_1pct_share"] < 1.0


def test_stationarity_interpretation_for_a_stationary_series(tmp_path):
    """White noise: ADF rejects a unit root, KPSS does not reject stationarity."""
    result = stationarity_tests(RNG.standard_normal(3000))
    assert result["adf_p"] < 0.05
    assert result["kpss_p"] > 0.05
    assert "stationary" in result["interpretation"].lower()


def test_stationarity_interpretation_for_a_random_walk(tmp_path):
    """A random walk: ADF fails to reject a unit root, KPSS rejects stationarity."""
    result = stationarity_tests(np.cumsum(RNG.standard_normal(3000)))
    assert result["adf_p"] > 0.05
    assert result["kpss_p"] < 0.05
    assert "non-stationary" in result["interpretation"].lower()


def test_acf_figure_reports_the_seasonal_lag(tmp_path):
    tone = np.sin(np.arange(3000) * 2 * np.pi / 144) + 0.05 * RNG.standard_normal(3000)
    stats = fig_acf_pacf(tone, nlags=288, path=tmp_path / "acf.png")
    assert stats["acf_144"] > 0.8        # strong daily correlation
    assert (tmp_path / "acf.png").exists()


def test_periodogram_recovers_a_known_period(tmp_path):
    """A 144-slot period is 24 h at 10-minute resolution."""
    tone = np.sin(np.arange(8928) * 2 * np.pi / 144)
    periods = fig_periodogram(tone, tmp_path / "psd.png")
    assert any(abs(p - 24.0) < 1.0 for p in periods)



