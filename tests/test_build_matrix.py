import numpy as np
import pytest
from datetime import date

from milan.config import load_config
from milan.etl.build_matrix import midnight_ms, raw_path, reduce_day
from milan.etl.manifest import fingerprint, load_manifest, save_manifest


KNOWN_MIDNIGHTS = {
    date(2013, 11, 4): 1383519600000,
    date(2013, 12, 16): 1387148400000,
    date(2014, 1, 1): 1388530800000,
}


@pytest.mark.parametrize("day,expected", KNOWN_MIDNIGHTS.items())
def test_midnight_ms_matches_file_timestamps(day, expected):
    assert midnight_ms(day, "Europe/Rome") == expected


def test_every_expected_raw_file_exists():
    cfg = load_config()
    missing = [d.isoformat() for d in cfg.dates() if not raw_path(cfg, d).exists()]
    assert missing == [], f"missing day files: {missing}"


def test_fingerprint_is_stable_and_discriminating():
    cfg = load_config()
    a = raw_path(cfg, date(2013, 12, 16))
    b = raw_path(cfg, date(2013, 12, 17))
    assert fingerprint(a) == fingerprint(a)
    assert fingerprint(a) != fingerprint(b)


def test_manifest_round_trips(tmp_path):
    p = tmp_path / "manifest.json"
    assert load_manifest(p) == {}          # absent file is an empty manifest
    save_manifest(p, {"2013-12-16": {"fingerprint": "x", "seconds": 1.0}})
    assert load_manifest(p)["2013-12-16"]["fingerprint"] == "x"


@pytest.mark.slow
def test_reduce_day_shape_dtype_and_golden_value():
    cfg = load_config()
    day = date(2013, 12, 16)
    grid = reduce_day(raw_path(cfg, day), midnight_ms(day, cfg.tz))
    assert grid.shape == (10000, 144)
    assert grid.dtype == np.float32
    assert grid[4159 - 1, 0] == pytest.approx(169.236305, rel=1e-6)
    assert grid[4159 - 1, 100] == pytest.approx(408.206177, rel=1e-6)


@pytest.mark.slow
def test_reduce_day_rejects_a_mismatched_date():
    cfg = load_config()
    day = date(2013, 12, 16)
    wrong = midnight_ms(date(2013, 12, 15), cfg.tz)
    with pytest.raises(RuntimeError, match="out of range"):
        reduce_day(raw_path(cfg, day), wrong)