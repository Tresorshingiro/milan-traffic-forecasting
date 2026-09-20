"""Phase 2: grid search over architecture hyperparameters.

Phase 1 fixed the data-level choices (input length, transform, loss, Huber delta).
Phase 2 searches only the architecture knobs and the learning rate. Selection is on
VALIDATION MAE; the test week is never consulted to choose anything.

Usage:
  python scripts/grid_search.py --L 144 --log1p --loss huber --huber-delta 0.3
  python scripts/grid_search.py --L 144 --log1p --loss huber --areas 4159
"""
from __future__ import annotations

import argparse
import json
from itertools import product

from milan.config import load_config
from milan.data.loader import target_areas
from milan.train import RunConfig, train_one

# 4 + 8 + 8 = 20 configurations per area.
GRIDS = {
    "dlinear": {"kernel": [25, 145]},
    "tcn": {"channels": [16, 32], "dropout": [0.0, 0.1]},
    "gru": {"hidden": [32, 64], "layers": [1, 2]},
}
LEARNING_RATES = [1e-3, 3e-4]


def expand(grid: dict) -> list[dict]:
    keys = sorted(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]


def completed(path) -> set[tuple[str, int]]:
    """(exp_id, area) pairs already in the ledger, so a crashed search can resume."""
    if not path.exists():
        return set()
    return {(r["exp_id"], int(r["area"])) for r in map(json.loads, path.open())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, required=True, help="from E2")
    parser.add_argument("--log1p", action="store_true", help="from E3")
    parser.add_argument("--loss", default="mse", choices=["mse", "huber"], help="from E4")
    parser.add_argument("--huber-delta", type=float, default=0.3,
                        help="standardised units; ~p95 of training residuals (see E3/E4)")
    parser.add_argument("--areas", type=int, nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_config()
    areas = args.areas or target_areas(cfg)

    runs = [
        RunConfig(
            exp_id=f"E6-{name}-{i:02d}", model=name, area=area, L=args.L,
            use_log1p=args.log1p, loss=args.loss, huber_delta=args.huber_delta,
            hyperparams=hyperparams,
            lr=lr,   # lr lives on RunConfig: model constructors do not accept it
            note=f"grid {name} {hyperparams} lr={lr}",
        )
        for area in areas
        for name in GRIDS
        for i, (hyperparams, lr) in enumerate(product(expand(GRIDS[name]), LEARNING_RATES))
    ]

    done = completed(cfg.ledger_path)
    todo = [r for r in runs if (r.exp_id, r.area) not in done]
    print(f"{len(runs)} runs over {len(areas)} area(s), {len(runs) - len(todo)} already "
          f"done -- selection on validation MAE only", flush=True)
    for run in todo:
        train_one(cfg, run, save_predictions=False, time_inference=False)


if __name__ == "__main__":
    main()
