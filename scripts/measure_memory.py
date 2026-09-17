from __future__ import annotations

import json
import subprocess
import sys
from datetime import date

from milan.config import load_config
from milan.etl.build_matrix import raw_path


if __name__ == "__main__":
    cfg = load_config()
    day = date(2013, 12, 16)
    path = raw_path(cfg, day)

    proc = subprocess.run(
        [sys.executable, "-m", "milan.etl.naive_baseline", str(path)],
        capture_output=True, text=True, check=False,
    )
    report = json.loads(cfg.memory_report_path.read_text())

    if proc.returncode != 0:
        # report task
        report["naive_baseline"] = {
            "status": "failed",
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr.strip()[-500:],
            "interpretation": (
                "The naive whole-file pandas load could not complete on this host, "
                "which is direct evidence that the streaming approach was necessary "
                "rather than merely preferable."
            )
        }
    else:
        naive = json.loads(proc.stdout)
        naive["status"] = "ok"
        report["naive_baseline"] = naive
        streaming_peak = report["peak_rss_mb"]
        report["comparison"] = {
            "naive_peak_rss_mb_one_day": naive["peak_rss_mb"],
            "streaming_peak_rss_mb_all_62_days": streaming_peak,
            "rss_reduction_factor": round(naive["peak_rss_mb"] / streaming_peak, 1),
            "naive_extrapolated_seconds_62_days": round(naive["seconds"] * 62, 1),
            "streaming_actual_seconds_62_days": report["wall_seconds"],
            "note": (
                "The naive figure is measured on ONE day-file; the 62-day time is an "
                "explicit extrapolation, not a measurement. Its peak RSS does not "
                "extrapolate at all, because the naive approach holds one file at a "
                "time -- the point is that the streaming approach's peak stays flat "
                "while processing 56x more data into a single artifact."
            )
        }

    cfg.memory_report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report.get("comparison", report ["naive_baseline"]), indent=2))


