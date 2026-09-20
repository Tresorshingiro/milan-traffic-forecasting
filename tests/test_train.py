import json
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from milan.config import load_config
from milan.train import RunConfig, append_ledger, set_seed, train_one


def test_set_seed_makes_model_init_reproducible():
    from milan.models.registry import build_model
    set_seed(42)
    a = [p.detach().clone() for p in build_model("gru", L=36).parameters()]
    set_seed(42)
    b = [p.detach().clone() for p in build_model("gru", L=36).parameters()]
    for pa, pb in zip(a, b):
        torch.testing.assert_close(pa, pb)


def test_append_ledger_writes_one_json_object_per_line(tmp_path):
    path = tmp_path / "experiments.jsonl"
    append_ledger(path, {"exp_id": "E1", "mae": 1.0})
    append_ledger(path, {"exp_id": "E2", "mae": 2.0})
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert [json.loads(line)["exp_id"] for line in lines] == ["E1", "E2"]


def test_training_loop_can_overfit_a_tiny_problem():
    """A sanity check on the loop itself, independent of the dataset.

    A model that cannot reduce its own training loss on 200 easy samples has a
    broken optimiser, loss, or shape somewhere.
    """
    from milan.models.registry import build_model
    set_seed(0)
    X = torch.randn(200, 36)
    y = X.sum(dim=1)                      # exactly learnable by a linear model
    model = build_model("dlinear", L=36)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(300):
        optimiser.zero_grad()
        loss = torch.nn.functional.mse_loss(model(X), y)
        loss.backward()
        optimiser.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first / 10, f"loss barely moved: {first:.4f} -> {loss.item():.4f}"


@pytest.mark.slow
def test_train_one_returns_every_documented_field(tmp_path):
    cfg = replace(load_config(), results=tmp_path)     # keep the real ledger clean
    run = RunConfig(model="dlinear", area=4159, L=36, exp_id="TEST",
                    note="smoke test", hyperparams={"kernel": 25})
    result = train_one(cfg, run, save_predictions=False)
    required = {
        "exp_id", "model", "area", "L", "use_log1p", "loss", "lr", "hyperparams",
        "seed", "n_parameters", "epochs_run", "best_epoch", "val_mae",
        "test_mae", "test_rmse", "test_mape", "test_smape", "test_mase", "test_skill",
        "train_seconds", "inference_ms_per_step", "peak_rss_mb", "timestamp", "note",
    }
    assert required <= set(result)
    assert result["epochs_run"] >= 1
    assert np.isfinite(result["test_mae"])


@pytest.mark.slow
def test_baselines_are_evaluated_on_the_same_test_window():
    from milan.train import evaluate_baselines
    cfg = load_config()
    result = evaluate_baselines(cfg, 4159)
    assert set(result) == {"persistence", "seasonal_naive"}
    for name, metrics in result.items():
        assert metrics["n_test"] == 1008, f"{name} scored {metrics['n_test']} points"
        assert np.isfinite(metrics["mae"])
    # Seasonal-naive should beat persistence on a strongly diurnal series... or not,
    # which is itself a finding. Just assert both produced a real number.
    assert result["seasonal_naive"]["mase"] > 0


@pytest.mark.slow
def test_train_one_can_skip_inference_timing(tmp_path):
    cfg = replace(load_config(), results=tmp_path)     # keep the real ledger clean
    run = RunConfig(model="dlinear", area=4159, L=36, exp_id="TEST-NOTIME",
                    hyperparams={"kernel": 25})
    result = train_one(cfg, run, save_predictions=False, time_inference=False)
    assert result["inference_ms_per_step"] is None
    assert result["train_seconds"] > 0
    assert (tmp_path / "experiments.jsonl").exists()
