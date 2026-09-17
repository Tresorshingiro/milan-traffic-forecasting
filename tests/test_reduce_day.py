import subprocess
import numpy as np
import pytest
from milan.config import load_config

SCRIPT = "src/milan/etl/reduce_day.awk"
DAY_FILE = "milan_dataset/sms-call-internet-mi-2013-12-16.txt"
T0 = 1387148400000

GOLDEN_4159_SLOT0 = 169.236305
GOLDEN_4159_SLOT100 = 408.206177
GOLDEN_GRAND_TOTAL = 88886170.1742

def _run(t0=T0, day_file=DAY_FILE):
    cfg = load_config()
    root = cfg.raw_dir.parent
    return subprocess.run(
        ["mawk", "-v", f"t0={t0}", "-f", str(root / SCRIPT), str(root / day_file)],
        capture_output= True, text=True, check=False,
    )

@pytest.fixture(scope="module")
def grid():
    proc = _run()
    assert proc.returncode == 0, proc.stderr
    arr = np.fromstring(proc.stdout, dtype=np.float64, sep=" ")
    assert arr.size == 10000 * 145
    return arr.reshape(10000, 145)

@pytest.mark.slow
def test_emits_every_square_in_ascending_order(grid):
    assert np.array_equal(grid[:, 0], np.arange(1, 10001))

@pytest.mark.slow
def test_matches_independently_computed_cell_values(grid):
    row = grid[grid[:, 0] == 4159][0]
    assert row[1] == pytest.approx(GOLDEN_4159_SLOT0, rel=1e-8)
    assert row[101] == pytest.approx(GOLDEN_4159_SLOT100, rel=1e-8)

@pytest.mark.slow
def test_grand_total_matches_direct_sum(grid):
    assert grid[:, 1:].sum() == pytest.approx(GOLDEN_GRAND_TOTAL, rel=1e-9)

@pytest.mark.slow
def test_no_negative_values(grid):
    assert grid[:, 1].min() >= 0.0

def test_missing_t0_is_a_hard_error():
    cfg = load_config()
    root = cfg.raw_dir.parent
    proc = subprocess.run(
        ["mawk", "-f", str(root / SCRIPT), "/dev/null"],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 2
    assert "t0" in proc.stderr

@pytest.mark.slow
def test_wrong_t0_is_a_hard_error():
    proc = _run(t0=T0 - 86_400_000)
    assert proc.returncode == 3
    assert "out of range" in proc.stderr
    assert proc.stdout.strip() == ""