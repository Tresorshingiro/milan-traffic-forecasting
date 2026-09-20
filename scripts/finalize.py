"""Select the best configuration per (model, area) and produce every reported artifact.

Selection is by VALIDATION MAE over all Phase-2 grid runs. The selected run is then
re-trained once with inference timing enabled and predictions saved, so the reported
timings and plots come from exactly the configuration the tables describe. Training
is deterministic, so the re-trained metrics must match the selected grid row.

Usage: python scripts/finalize.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from milan.config import load_config
from milan.data.loader import target_areas
from milan.plotting import plot_all_models_zoom, plot_predictions, setup_style
from milan.train import RunConfig, evaluate_baselines, train_one

METRICS = ["mae", "rmse", "mape", "smape", "mase", "skill"]
MODELS = ("dlinear", "tcn", "gru")


def load_ledger(path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(line) for line in path.open()])


def h1_holds(predicted: str, actual: str) -> bool:
    """prediction.json names either 'dlinear' or the nonlinear class 'tcn_or_gru'."""
    return actual == "dlinear" if predicted == "dlinear" else actual in ("tcn", "gru")


if __name__ == "__main__":
    setup_style()
    cfg = load_config()
    tables, figures = cfg.results / "tables", cfg.results / "figures"
    tables.mkdir(parents=True, exist_ok=True)

    ledger = load_ledger(cfg.ledger_path)
    grid = ledger[ledger["exp_id"].str.startswith("E6-")].copy()
    if grid.empty:
        raise SystemExit("no Phase-2 runs found -- run scripts/grid_search.py first")

    areas = target_areas(cfg)
    rows, timing_rows, forecasts_by_area = [], [], {}

    for area in areas:
        forecasts_by_area[area] = {}

        # Baselines first: the reference every model is judged against.
        for name, metrics in evaluate_baselines(cfg, area).items():
            rows.append({"area": area, "model": name,
                         **{m: metrics[m] for m in METRICS}, "n_parameters": 0})

        for model in MODELS:
            subset = grid[(grid["area"] == area) & (grid["model"] == model)]
            if subset.empty:
                print(f"WARNING: no grid runs for {model} on area {area}; skipping")
                continue
            best = subset.loc[subset["val_mae"].idxmin()]

            run = RunConfig(
                exp_id=f"FINAL-{model}-{area}", model=model, area=area,
                L=int(best["L"]), use_log1p=bool(best["use_log1p"]),
                loss=str(best["loss"]), huber_delta=float(best["huber_delta"]),
                lr=float(best["lr"]), hyperparams=dict(best["hyperparams"]),
                note=f"final: {best['exp_id']} selected from {len(subset)} grid runs "
                     f"by val MAE",
            )
            record = train_one(cfg, run, save_predictions=True, time_inference=True)
            if not np.isclose(record["val_mae"], best["val_mae"]):
                print(f"WARNING: {run.exp_id} val MAE {record['val_mae']:.4f} does not "
                      f"reproduce the grid row {best['val_mae']:.4f}")

            rows.append({"area": area, "model": model,
                         **{m: record[f"test_{m}"] for m in METRICS},
                         "n_parameters": record["n_parameters"]})
            timing_rows.append({
                "area": area, "model": model,
                "n_parameters": record["n_parameters"],
                "epochs_run": record["epochs_run"],
                "best_epoch": record["best_epoch"],
                "train_seconds": record["train_seconds"],
                "inference_ms_per_step": record["inference_ms_per_step"],
                "lr": record["lr"],
                "config": json.dumps(record["hyperparams"]),
            })

            forecast = np.load(cfg.results / "predictions" /
                               f"{run.exp_id}_{model}_{area}.npy")
            forecasts_by_area[area][model] = forecast
            plot_predictions(cfg, area, model, forecast,
                             figures / f"07_pred_{model}_{area}.png")

        # One zoomed 24 h panel per area: Dec 16 is day_offset 0 (a Monday).
        if forecasts_by_area[area]:
            plot_all_models_zoom(cfg, area, forecasts_by_area[area], 0,
                                 figures / f"08_zoom_{area}.png")

    results = pd.DataFrame(rows)
    results.to_csv(tables / "results_all.csv", index=False)
    for area in areas:
        table = results[results["area"] == area].drop(columns=["area"])
        table.to_csv(tables / f"results_area_{area}.csv", index=False)
        print(f"\n=== Area {area} ===")
        print(table.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    pd.DataFrame(timing_rows).to_csv(tables / "timing.csv", index=False)

    # --- H1: did the pre-registered prediction hold? ----------------------
    prediction = json.loads((tables / "prediction.json").read_text())
    trained = results[results["model"].isin(MODELS)]
    actual_winner = {
        int(area): str(group.loc[group["mase"].idxmin(), "model"])
        for area, group in trained.groupby("area")
    }
    h1_match = {
        area: h1_holds(prediction["predicted_winner_by_area"][str(area)], winner)
        for area, winner in actual_winner.items()
    }

    # --- H2: does raw MAE mislead ACROSS areas? ----------------------------
    # MASE = MAE / (a per-area constant), so within one area the two metrics always
    # rank models identically. H2 is a cross-area claim: for each model, rank the
    # three areas from easiest to hardest by MAE and by MASE, and compare.
    h2 = {}
    for model, group in results.groupby("model"):
        by_mae = [int(a) for a in group.sort_values("mae")["area"]]
        by_mase = [int(a) for a in group.sort_values("mase")["area"]]
        h2[model] = {"easiest_first_by_mae": by_mae,
                     "easiest_first_by_mase": by_mase,
                     "agree": by_mae == by_mase}

    persistence_mae = results[results["model"] == "persistence"].set_index("area")["mae"]
    verdict = {
        "H1_predicted_winner_by_area": prediction["predicted_winner_by_area"],
        "H1_actual_winner_by_area": actual_winner,
        "H1_match_by_area": h1_match,
        "H1_predicted_predictability_order": prediction["most_to_least_predictable"],
        "H2_area_rankings_by_model": h2,
        "H2_supported": any(not v["agree"] for v in h2.values()),
        "models_beating_seasonal_naive": {
            int(area): sorted(group[group["skill"] > 0]["model"])
            for area, group in trained.groupby("area")
        },
        # Persistence is the demanding bar here: seasonal-naive sits near MASE 1.
        "models_beating_persistence": {
            int(area): sorted(group[group["mae"] < persistence_mae[area]]["model"])
            for area, group in trained.groupby("area")
        },
    }
    (tables / "hypothesis_test.json").write_text(json.dumps(verdict, indent=2))
    print("\n=== HYPOTHESIS VERDICT ===")
    print(json.dumps(verdict, indent=2))
