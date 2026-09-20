"""Failure analysis over the final forecasts. Usage: python scripts/failure_analysis.py"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from milan.analysis.failure import (
    error_by_day, error_by_slot, phase_lag_diagnostic,
    residual_autocorrelation, worst_windows,
)
from milan.config import load_config
from milan.data.loader import load_area, target_areas
from milan.models.baselines import persistence
from milan.plotting import MODEL_COLORS, save, setup_style

MODELS = ("dlinear", "tcn", "gru")

if __name__ == "__main__":
    setup_style()
    cfg = load_config()
    figures, tables = cfg.results / "figures", cfg.results / "tables"
    test = cfg.split_cols("test")
    report = {}

    for area in target_areas(cfg):
        series = load_area(cfg, area)
        truth = series[test.start: test.stop]
        baseline = persistence(series, test.start, test.stop)

        per_slot, per_day, area_report = {}, {}, {}
        for model in MODELS:
            path = cfg.results / "predictions" / f"FINAL-{model}-{area}_{model}_{area}.npy"
            if not path.exists():
                print(f"WARNING: {path.name} missing; run scripts/finalize.py first")
                continue
            forecast = np.load(path)
            per_slot[model] = error_by_slot(cfg, truth, forecast)
            per_day[model] = error_by_day(cfg, truth, forecast)
            area_report[model] = {
                "residual_acf": residual_autocorrelation(truth, forecast),
                "phase_lag": phase_lag_diagnostic(truth, forecast),
                "worst_windows": worst_windows(cfg, truth, forecast, n=10),
                "per_day_mae": per_day[model].tolist(),
            }

        # The persistence baseline gets the same treatment, as the reference for
        # "has this model actually learned anything beyond repeating the last value?"
        per_slot["persistence"] = error_by_slot(cfg, truth, baseline)
        per_day["persistence"] = error_by_day(cfg, truth, baseline)
        area_report["persistence"] = {
            "residual_acf": residual_autocorrelation(truth, baseline),
            "phase_lag": phase_lag_diagnostic(truth, baseline),
            "per_day_mae": error_by_day(cfg, truth, baseline).tolist(),
        }
        report[area] = area_report

        if len(per_slot) > 1:           # at least one model besides persistence
            hours = np.arange(cfg.slots_per_day) / 6.0
            fig, ax = plt.subplots(figsize=(9, 3.2))
            for model, values in per_slot.items():
                ax.plot(hours, values, label=model, color=MODEL_COLORS.get(model))
            ax.set(xlabel="Hour of day", ylabel="Mean absolute error",
                   title=f"Error by time of day, square {area} (Dec 16-22)",
                   xticks=range(0, 25, 3))
            ax.legend(ncols=3)
            save(fig, figures / f"09_error_by_hour_{area}.png")

            dates = [str(cfg.dates()[cfg.test_days[0] + d]) for d in range(7)]
            fig, ax = plt.subplots(figsize=(7, 3.2))
            width = 0.8 / len(per_day)
            for i, (model, values) in enumerate(per_day.items()):
                ax.bar(np.arange(7) + i * width, values, width, label=model,
                       color=MODEL_COLORS.get(model))
            ax.set(xlabel="Test day", ylabel="Mean absolute error",
                   title=f"Error by test day, square {area}")
            ax.set_xticks(np.arange(7) + 0.4 - width / 2)
            ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=7)
            ax.legend(ncols=3)
            save(fig, figures / f"10_error_by_day_{area}.png")

    (tables / "failure_analysis.json").write_text(json.dumps(report, indent=2, default=str))

    print("\n=== PHASE LAG (best_shift > 0 means the model is lagging reality) ===")
    for area, models in report.items():
        for model, data in models.items():
            lag = data["phase_lag"]
            acf1 = data["residual_acf"]["acf_1"]
            print(f"area {area:5d}  {model:14s} best_shift={lag['best_shift']}  "
                  f"residual_acf_1={acf1:+.3f}")
