import numpy as np
import pytest

from milan.config import load_config
from milan.data.loader import (Scaler, fit_scaler, load_area, load_square_totals, make_windows, open_matrix, target_areas, top_areas,)


def test_window_for_target_t_ends_at_t_minus_one():
    series = np.arange(100, dtype=np.float64)
    X, y = make_windows(series, L=5, start=10, end=13)
    assert X.shape == (3, 5) and y.shape == (3,)
    np.testing.assert_array_equal(X[0], [5, 6, 7, 8, 9])
    assert y[0] == 10
    np.testing.assert_array_equal(X[2], [7, 8, 9, 10, 11])
    assert y[2] == 12

def test_no_window_contains_its_own_target_or_the_future():
    series = np.arange(500, dtype=np.float64)
    X, y = make_windows(series, L=20, start=100, end=200)
    for i in range(len(y)):
        assert y[i] not in X[i], "target leaked into its own input window"
        assert X[i].max() < y[i], "input window contains a future value"


def test_window_count_matches_the_documented_split_sizes():
    cfg = load_config()
    series = np.arange(cfg.n_steps, dtype=np.float64)
    L = 144
    train = cfg.split_cols("train")
    X, y = make_windows(series, L, train.start + L, train.stop)
    assert len(y) == 5472 - L == 5328
    for name, expected in (("val", 1008), ("test", 1008)):
        s = cfg.split_cols(name)
        _, yy = make_windows(series, L, s.start, s.stop)
        assert len(yy) == expected


def test_insufficient_history_is_rejected():
    series = np.arange(100, dtype=np.float64)
    with pytest.raises(ValueError, match="history"):
        make_windows(series, L=20, start=10, end=50)



def test_scaler_uses_only_the_values_it_was_fitted_on():
    train = np.array([1.0, 2.0, 3.0, 4.0])
    scaler = fit_scaler(train, use_log1p=False)
    assert scaler.mu == pytest.approx(2.5)
    assert scaler.sigma == pytest.approx(np.std(train))
    # An unseen, much larger value must not change the statistics.
    scaler.transform(np.array([1000.0]))
    assert scaler.mu == pytest.approx(2.5)


def test_scaler_round_trips():
    values = np.array([0.0, 1.5, 10.0, 250.0])
    for use_log1p in (False, True):
        scaler = fit_scaler(values, use_log1p=use_log1p)
        np.testing.assert_allclose(scaler.inverse(scaler.transform(values)), values, rtol=1e-9)


def test_inverse_clamps_to_non_negative_traffic():
    scaler = fit_scaler(np.array([1.0, 2.0, 3.0]), use_log1p=False)
    assert scaler.inverse(np.array([-50.0]))[0] == 0.0


def test_zero_variance_training_split_is_rejected():
    with pytest.raises(ValueError, match="variance"):
        fit_scaler(np.array([7.0, 7.0, 7.0]), use_log1p=False)



@pytest.mark.slow
def test_matrix_has_the_documented_shape_and_dtype():
    cfg = load_config()
    matrix = open_matrix(cfg)
    assert matrix.shape == (10000, 8928)
    assert matrix.dtype == np.float32


@pytest.mark.slow
def test_load_area_matches_the_etl_golden_value():
    """Square 4159, Dec 16 slot 0 -> column 6480. Same golden value as Task 2."""
    cfg = load_config()
    series = load_area(cfg, 4159)
    assert series.shape == (8928,)
    assert series[6480] == pytest.approx(169.236305, rel=1e-6)


@pytest.mark.slow
def test_load_area_rejects_out_of_range_ids():
    cfg = load_config()
    for bad in (0, 10001, -1):
        with pytest.raises(ValueError, match="square_id"):
            load_area(cfg, bad)


@pytest.mark.slow
def test_top_areas_are_descending_and_distinct():
    cfg = load_config()
    totals = load_square_totals(cfg)
    areas = top_areas(cfg, 3)
    assert len(set(areas)) == 3
    values = [totals[a - 1] for a in areas]
    assert values == sorted(values, reverse=True)
    assert all(1 <= a <= 10000 for a in areas)


@pytest.mark.slow
def test_target_areas_are_the_three_study_areas():
    cfg = load_config()
    areas = target_areas(cfg)
    assert len(areas) == 3
    assert areas[0] == top_areas(cfg, 1)[0]


