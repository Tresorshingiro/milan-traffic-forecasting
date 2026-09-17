from __future__ import annotations

import json
import resource
import sys
import time

import pandas as pd

COLUMNS = ["square_id", "time_ms", "country", "sms_in", "sms_out", "call_in", "call_out", "internet"]

def main(path: str) -> None:
    started = time.perf_counter()
    frame = pd.read_csv(path, sep="\t", header=None, names=COLUMNS)
    elapsed = time.perf_counter() - started

    series = frame.groupby(["square_id", "time_ms"])["internet"].sum()

    print(json.dumps({
        "approach": "naive_pandas_whole_file_all_columns",
        "file": path,
        "rows": int(len(frame)),
        "frame_mb": round(frame.memory_usage(deep=True).sum() / 1e6, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        "seconds": round(elapsed, 2),
        "reduced_cells": int(len(series)),
    }))

if __name__ == "__main__":
    main(sys.argv[1])