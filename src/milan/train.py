from __future__ import annotations

import json
import random
import resource
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from milan.config import Config
from milan.data.loader import fit_scaler, load_area, make_windows
from milan.evaluate import all_metrics, mae, seasonal_naive_mae_insample
from milan.models.baselines import persistence, seasonal_naive
from milan.models.registry import build_model, count_parameters

LOSSES = {"mse": nn.MSELoss, "huber": nn.HuberLoss}


@dataclass(frozen=True)
class RunConfig:
    model: str
    area: int
    L: int = 144
    use_log1p: bool = False
    loss: str = "mse"
    huber_delta: float = 1.0      # in standardised units; only used when loss == "huber"
    lr: float = 1e-3
    hyperparams: dict = field(default_factory=dict)
    seed: int = 42
    exp_id: str = "adhoc"
    note: str = ""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def append_ledger(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def evaluate_baselines(cfg: Config, area: int) -> dict[str, dict]:
    """Score persistence and seasonal-naive on the same test window as the models."""
    series = load_area(cfg, area)
    train, test = cfg.split_cols("train"), cfg.split_cols("test")
    denom = seasonal_naive_mae_insample(series[train], m=cfg.slots_per_day)
    truth = series[test.start: test.stop]

    results = {}
    for name, forecast in (
        ("persistence", persistence(series, test.start, test.stop)),
        ("seasonal_naive", seasonal_naive(series, test.start, test.stop, m=cfg.slots_per_day)),
    ):
        metrics = all_metrics(truth, forecast, denom)
        metrics["n_test"] = int(truth.size)
        results[name] = metrics
    return results


def train_one(cfg: Config, run: RunConfig, save_predictions: bool = True) -> dict:
    """Train one model on one area and score it on the test week.

    Returns the ledger record and appends it to results/experiments.jsonl.
    """
    torch.set_num_threads(4)          # fixed, so timings are comparable across runs
    set_seed(run.seed)

    series = load_area(cfg, run.area)
    train, val, test = (cfg.split_cols(s) for s in ("train", "val", "test"))

    # Scaler statistics come from the training split ONLY.
    scaler = fit_scaler(series[train], use_log1p=run.use_log1p)
    scaled = scaler.transform(series)

    X_train, y_train = make_windows(scaled, run.L, train.start + run.L, train.stop)
    X_val, y_val = make_windows(scaled, run.L, val.start, val.stop)
    X_test, y_test = make_windows(scaled, run.L, test.start, test.stop)

    truth_val = series[val.start: val.stop]
    truth_test = series[test.start: test.stop]
    denom = seasonal_naive_mae_insample(series[train], m=cfg.slots_per_day)

    model = build_model(run.model, L=run.L, **run.hyperparams)
    optimiser = torch.optim.Adam(model.parameters(), lr=run.lr)
    criterion = (LOSSES["huber"](delta=run.huber_delta) if run.loss == "huber"
                 else LOSSES[run.loss]())

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=cfg.batch_size, shuffle=True,   # windows, not splits
    )
    val_tensor = torch.from_numpy(X_val)

    best_val, best_epoch, best_state, epochs_run = float("inf"), -1, None, 0
    started = time.perf_counter()

    for epoch in range(cfg.max_epochs):
        model.train()
        for xb, yb in loader:
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
            optimiser.step()

        model.eval()
        with torch.no_grad():
            predicted = scaler.inverse(model(val_tensor).numpy())
        current = mae(truth_val, predicted)       # early stopping in ORIGINAL units
        epochs_run = epoch + 1

        if current < best_val:
            best_val, best_epoch = current, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= cfg.patience:
            break

    train_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)             # restore the best weights, not the last
    model.eval()

    with torch.no_grad():
        forecast = scaler.inverse(model(torch.from_numpy(X_test)).numpy())

    # Inference cost: median over 3 repeats of one full single-step pass over the week.
    timings = []
    for _ in range(3):
        t0 = time.perf_counter()
        with torch.no_grad():
            for i in range(len(X_test)):
                model(torch.from_numpy(X_test[i: i + 1]))
        timings.append((time.perf_counter() - t0) / len(X_test) * 1000.0)

    metrics = all_metrics(truth_test, forecast, denom)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **{k: v for k, v in asdict(run).items()},
        "n_parameters": count_parameters(model),
        "n_train_windows": int(len(y_train)),
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "val_mae": best_val,
        "test_mae": metrics["mae"],
        "test_rmse": metrics["rmse"],
        "test_mape": metrics["mape"],
        "test_mape_excluded": metrics["mape_excluded"],
        "test_smape": metrics["smape"],
        "test_mase": metrics["mase"],
        "test_skill": metrics["skill"],
        "mase_denominator": denom,
        "train_seconds": round(train_seconds, 2),
        "inference_ms_per_step": round(float(np.median(timings)), 4),
        "peak_rss_mb": round(_peak_rss_mb(), 1),
    }
    append_ledger(cfg.ledger_path, record)

    if save_predictions:
        out = cfg.results / "predictions"
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / f"{run.exp_id}_{run.model}_{run.area}.npy", forecast)

    print(f"{run.exp_id:6s} {run.model:8s} area={run.area:5d} "
          f"val_mae={best_val:8.3f} test_mae={metrics['mae']:8.3f} "
          f"MASE={metrics['mase']:.3f} skill={metrics['skill']:+.3f} "
          f"{train_seconds:5.1f}s")
    return record

