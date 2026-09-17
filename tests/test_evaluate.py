import numpy as np
import pytest

from milan.evaluate import (
    all_metrics, mae, mape, mase, rmse, seasonal_naive_mae_insample,
    skill_score, smape,
)
from milan.models.baselines import persistence, seasonal_naive


def test_mae_and_rmse_hand_computed():
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([2.0, 2.0, 2.0])
    assert mae(y, yhat) == pytest.approx(2 / 3)
    assert rmse(y, yhat) == pytest.approx(np.sqrt(2 / 3))


def test_perfect_forecast_scores_zero_error():
    y = np.array([5.0, 10.0, 15.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert smape(y, y) == 0.0
    assert mape(y, y)[0] == 0.0


def test_mase_is_mae_relative_to_the_seasonal_naive_denominator():
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([2.0, 2.0, 2.0])
    assert mase(y, yhat, denom=2.0) == pytest.approx((2 / 3) / 2)


def test_mase_of_one_means_no_better_than_seasonal_naive():
    assert skill_score(1.0) == pytest.approx(0.0)
    assert skill_score(0.5) == pytest.approx(0.5)   # half the naive error
    assert skill_score(2.0) == pytest.approx(-1.0)  # twice as bad as naive


def test_seasonal_naive_denominator_hand_computed():
    # m=2: |3-1| + |4-2| = 4, over 2 differences -> 2.0
    series = np.array([1.0, 2.0, 3.0, 4.0])
    assert seasonal_naive_mae_insample(series, m=2) == pytest.approx(2.0)


def test_mape_excludes_near_zero_targets_and_reports_the_count():
    y = np.array([100.0, 0.0, 0.0, 200.0])
    yhat = np.array([110.0, 5.0, 5.0, 180.0])
    value, excluded = mape(y, yhat, threshold=1e-3)
    assert excluded == 2
    # mean(|10|/100, |20|/200) * 100 = mean(10%, 10%) = 10%
    assert value == pytest.approx(10.0)


def test_mape_returns_nan_when_every_target_is_excluded():
    y = np.zeros(4)
    value, excluded = mape(y, np.ones(4), threshold=1e-3)
    assert excluded == 4
    assert np.isnan(value)


def test_smape_is_bounded_and_symmetric():
    y = np.array([100.0]); over = np.array([150.0]); under = np.array([50.0])
    assert smape(y, np.array([0.0])) == pytest.approx(200.0)   # upper bound
    assert 0 < smape(y, over) < 200 and 0 < smape(y, under) < 200
    assert smape(np.zeros(3), np.zeros(3)) == 0.0              # 0/0 defined as 0


def test_baselines_use_the_documented_indices():
    series = np.arange(1000, dtype=np.float64)
    np.testing.assert_array_equal(persistence(series, 500, 503), [499, 500, 501])
    np.testing.assert_array_equal(seasonal_naive(series, 500, 503, m=144), [356, 357, 358])


def test_baselines_reject_insufficient_history():
    series = np.arange(100, dtype=np.float64)
    with pytest.raises(ValueError, match="history"):
        seasonal_naive(series, 10, 50, m=144)
    with pytest.raises(ValueError, match="history"):
        persistence(series, 0, 50)


def test_persistence_on_a_constant_series_is_exact():
    series = np.full(500, 7.0)
    pred = persistence(series, 200, 300)
    assert mae(series[200:300], pred) == 0.0


def test_all_metrics_returns_every_documented_key():
    y = np.array([10.0, 20.0, 30.0])
    yhat = np.array([12.0, 18.0, 33.0])
    result = all_metrics(y, yhat, denom=5.0)
    assert set(result) == {"mae", "rmse", "mape", "mape_excluded", "smape", "mase", "skill"}
    assert result["mase"] == pytest.approx(result["mae"] / 5.0)
    assert result["skill"] == pytest.approx(1 - result["mase"])

