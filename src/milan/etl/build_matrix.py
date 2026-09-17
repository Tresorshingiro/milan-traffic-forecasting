from __future__ import annotations
import json
import resource
import subprocess
import time
from datetime import date, datetime
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

import numpy as np

from milan.config import Config, load_config
from milan.etl.manifest import fingerprint, load_manifest, save_manifest

AWK_SCRIPT = Path(__file__).with_name("reduce_day.awk")
_MS_PER_SLOT = 600_000


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def midnight_ms(day: date, tz: str) -> int:
    """Local midnight for `day` as epoch milliseconds."""
    return int(datetime(day.year, day.month, day.day, tzinfo=ZoneInfo(tz)).timestamp() * 1000)


def raw_path(cfg: Config, day: date) -> Path:
    return cfg.raw_dir / f"sms-call-internet-mi-{day.isoformat()}.txt"


def reduce_day(path: Path, t0: int) -> np.ndarray:
    """Run the awk reducer over one day-file and return a (10000, 144) float32 grid."""
    proc = subprocess.run(
        ["mawk", "-v", f"t0={t0}", "-f", str(AWK_SCRIPT), str(path)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"reducer failed on {path.name} (rc={proc.returncode}): {proc.stderr.strip()}"
        )
    flat = np.fromstring(proc.stdout, dtype=np.float32, sep=" ")
    expected = 10000 * 145
    if flat.size != expected:
        raise RuntimeError(f"{path.name}: parsed {flat.size} values, expected {expected}")
    table = flat.reshape(10000, 145)
    if not np.array_equal(table[:, 0], np.arange(1, 10001, dtype=np.float32)):
        raise RuntimeError(f"{path.name}: square ids are not 1..10000 in ascending order")
    return np.ascontiguousarray(table[:, 1:])


def build(cfg: Config | None = None, force: bool = False) -> dict:
    """Build the traffic matrix, totals vector, manifest, and memory report."""
    cfg = cfg or load_config()
    cfg.artifacts.mkdir(parents=True, exist_ok=True)
    manifest = {} if force else load_manifest(cfg.manifest_path)

    if force or not cfg.matrix_path.exists():
        matrix = np.lib.format.open_memmap(
            cfg.matrix_path, mode="w+", dtype=np.float32,
            shape=(cfg.n_squares, cfg.n_steps),
        )
    else:
        matrix = np.lib.format.open_memmap(cfg.matrix_path, mode="r+")

    wall_start = time.perf_counter()
    per_day: list[float] = []
    skipped = 0

    for index, day in enumerate(cfg.dates()):
        path = raw_path(cfg, day)
        key = day.isoformat()
        fp = fingerprint(path)
        if manifest.get(key, {}).get("fingerprint") == fp:
            skipped += 1
            continue

        started = time.perf_counter()
        grid = reduce_day(path, midnight_ms(day, cfg.tz))
        matrix[:, cfg.day_cols(index, index + 1)] = grid
        matrix.flush()
        elapsed = time.perf_counter() - started

        manifest[key] = {
            "fingerprint": fp,
            "seconds": round(elapsed, 3),
            "sum": float(grid.sum()),
            "nonzero_frac": round(float((grid > 0).mean()), 6),
        }
        save_manifest(cfg.manifest_path, manifest)
        per_day.append(elapsed)
        del grid
        print(f"[{index + 1:2d}/{cfg.n_days}] {key}  {elapsed:5.1f}s  "
              f"peak_rss={peak_rss_mb():6.1f} MB")

    # Per-area totals, accumulated one day-block at a time to bound memory.
    totals = np.zeros(cfg.n_squares, dtype=np.float64)
    for col in range(0, cfg.n_steps, cfg.slots_per_day):
        totals += matrix[:, col:col + cfg.slots_per_day].sum(axis=1, dtype=np.float64)
    np.save(cfg.totals_path, totals)

    raw_bytes = sum(raw_path(cfg, d).stat().st_size for d in cfg.dates())
    matrix_bytes = cfg.matrix_path.stat().st_size
    report = {
        "approach": "streaming_awk_reduction_to_float32_memmap",
        "days_processed": len(per_day),
        "days_skipped_idempotent": skipped,
        "wall_seconds": round(time.perf_counter() - wall_start, 1),
        "median_seconds_per_day": round(median(per_day), 2) if per_day else None,
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "raw_bytes": raw_bytes,
        "matrix_bytes": matrix_bytes,
        "reduction_factor": round(raw_bytes / matrix_bytes, 1),
        "matrix_shape": [cfg.n_squares, cfg.n_steps],
        "dtype": "float32",
    }
    cfg.memory_report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report

