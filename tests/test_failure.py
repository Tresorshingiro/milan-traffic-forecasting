import numpy as np
import pytest

from milan.analysis.failure import (
    error_by_day, error_by_slot, phase_lag_diagnostic,
    residual_autocorrelation, worst_windows,
)
from milan.config import load_config

RNG = np.random.default_rng(0)


def test_phase_lag_diagnostic_detects_a_pure_one_step_lag():
    """A forecast that is just yesterday-shifted reality must be flagged."""
    truth = np.sin(np.arange(1008) * 2 * np.pi / 144) * 100 + 500
    lagging = np.concatenate([[truth[0]], truth[:-1]])     # forecast = truth shifted late
    result = phase_lag_diagnostic(truth, lagging, max_shift=3)
    assert result["best_shift"] == 1, result["mae_by_shift"]
    assert result["mae_by_shift"][1] < result["mae_by_shift"][0]


def test_phase_lag_diagnostic_reports_no_shift_for_an_unbiased_forecast():
    truth = np.sin(np.arange(1008) * 2 * np.pi / 144) * 100 + 500
    unbiased = truth + RNG.standard_normal(1008) * 2.0
    assert phase_lag_diagnostic(truth, unbiased, max_shift=3)["best_shift"] == 0


def test_residual_autocorrelation_is_high_for_a_lagging_forecast():
    truth = np.sin(np.arange(1008) * 2 * np.pi / 144) * 100 + 500
    lagging = np.concatenate([[truth[0]], truth[:-1]])
    result = residual_autocorrelation(truth, lagging, lags=(1, 2, 144))
    assert set(result) == {"acf_1", "acf_2", "acf_144"}
    assert abs(result["acf_144"]) > 0.5     # the residual inherits the daily cycle


def test_residual_autocorrelation_is_low_for_white_noise_errors():
    truth = np.full(1008, 500.0)
    result = residual_autocorrelation(truth, truth + RNG.standard_normal(1008), lags=(1,))
    assert abs(result["acf_1"]) < 0.1


def test_error_by_slot_has_one_value_per_daily_slot():
    cfg = load_config()
    truth = RNG.random(1008) * 100
    errors = error_by_slot(cfg, truth, truth + 5.0)
    assert errors.shape == (144,)
    np.testing.assert_allclose(errors, 5.0)


def test_error_by_day_has_one_value_per_test_day():
    cfg = load_config()
    truth = RNG.random(1008) * 100
    per_day = error_by_day(cfg, truth, truth + 3.0)
    assert per_day.shape == (7,)
    np.testing.assert_allclose(per_day, 3.0)


def test_error_by_day_isolates_a_single_bad_day():
    cfg = load_config()
    truth = np.zeros(1008)
    forecast = np.zeros(1008)
    forecast[5 * 144:6 * 144] = 10.0        # day index 5 is Dec 21
    per_day = error_by_day(cfg, truth, forecast)
    assert per_day[5] == pytest.approx(10.0)
    assert per_day[[0, 1, 2, 3, 4, 6]].max() == 0.0


def test_worst_windows_are_ordered_and_dated():
    cfg = load_config()
    truth = np.zeros(1008)
    forecast = np.zeros(1008)
    forecast[100] = 50.0
    forecast[200] = 90.0
    worst = worst_windows(cfg, truth, forecast, n=3)
    assert [w["index"] for w in worst][:2] == [200, 100]
    assert worst[0]["abs_error"] == pytest.approx(90.0)
    assert "timestamp" in worst[0] and "2013-12" in worst[0]["timestamp"]
