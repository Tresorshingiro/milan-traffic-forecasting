from datetime import date
from milan.config import load_config

def test_grid_dimensions():
    cfg = load_config()
    assert cfg.n_squares == 10000
    assert cfg.slots_per_day == 144
    assert cfg.n_days == 62
    assert cfg.n_steps == 8928

def test_split_columns_grid_without_overlap():
    cfg = load_config()
    train, val, test = cfg.split_cols("train"), cfg.split_cols("val"), cfg.split_cols("test")
    assert (train.start, train.stop) == (0, 5472)
    assert (val.start, val.stop) == (5472, 6480)
    assert (test.start, test.stop) == (6480, 7488)
    assert train.stop == val.start and val.stop == test.start
    assert cfg.n_steps - test.stop == 1440

def test_dates_span_the_documented_range():
    cfg = load_config()
    days = cfg.dates()
    assert len(days) == 62
    assert days[0] == date(2013, 11, 1)
    assert days[-1] == date(2014, 1, 1)

def test_test_week_is_monday_to_sunday():
    cfg = load_config()
    days = cfg.dates()
    d0, d1 = cfg.test_days
    assert days[d0] == date(2013, 12, 16) and days[d0].weekday() == 0
    assert days[d1 - 1] == date(2013, 12, 22) and days[d1 - 1].weekday() == 6

