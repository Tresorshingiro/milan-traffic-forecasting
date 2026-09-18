from __future__ import annotations

import json

from milan.analysis.eda import (
    fig_acf_pacf, fig_area_series, fig_lorenz, fig_periodogram,
    fig_stl, fig_traffic_distribution, stationarity_tests,
)
from milan.config import load_config
from milan.data.loader import load_area, load_square_totals, top_areas
from milan.plotting import setup_style

if __name__ == "__main__":
    setup_style()
    cfg = load_config()
    figures = cfg.results / "figures"
    tables = cfg.results / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    totals = load_square_totals(cfg)
    summary = {"distribution": fig_traffic_distribution(totals, figures / "01_distribution.png")}
    summary["gini"] = fig_lorenz(totals, figures / "02_lorenz.png")

    top3 = top_areas(cfg, 3)
    summary["top3_areas"] = top3
    summary["required_areas"] = top3 + list(cfg.extra_areas)

    # Task 2 item 3: the five required areas over the first two weeks (Nov 1-14).
    fig_area_series(cfg, summary["required_areas"], 0, 14, figures / "03_five_areas_2weeks.png")

    # Task 2 item 4: deeper analysis of the highest-traffic area, on the training split only.
    busiest = top3[0]
    train = cfg.split_cols("train")
    series = load_area(cfg, busiest)[train]
    summary["busiest_area"] = busiest
    summary["seasonal_strength"] = fig_stl(series, cfg.slots_per_day, figures / "04_stl.png")
    summary["acf"] = fig_acf_pacf(series, nlags=288, path=figures / "05_acf_pacf.png")
    summary["stationarity"] = stationarity_tests(series)
    summary["dominant_periods_hours"] = fig_periodogram(series, figures / "06_periodogram.png")

    (tables / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


