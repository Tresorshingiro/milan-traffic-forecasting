from __future__ import annotations

import argparse

from milan.config import load_config
from milan.data.loader import target_areas
from milan.train import RunConfig, train_one


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["E1", "E2", "E3", "E4", "E5"])
    parser.add_argument("--L", type=int, default=144,
                        help="input length; set from what E2 selected")
    parser.add_argument("--log1p", action="store_true",
                        help="use log1p preprocessing; set from what E3 selected")
    parser.add_argument("--loss", default="mse", choices=["mse", "huber"],
                        help="loss; set from what E4 selected")
    parser.add_argument("--huber-delta", type=float, default=0.3,
                        help="Huber threshold in standardised units; ~p95 of training "
                             "residuals under log1p (see E3 in the experiment log)")
    args = parser.parse_args()

    cfg = load_config()
    area = target_areas(cfg)[0]          # highest-traffic area
    runs: list[RunConfig] = []

    if args.stage == "E1":
        # Reference point. Sensible defaults, no tuning, to establish where we start
        # relative to the two baselines.
        runs = [RunConfig(exp_id="E1", model="gru", area=area, L=144,
                          hyperparams={"hidden": 64, "layers": 1},
                          note="reference GRU; L=144 from ACF daily peak")]

    elif args.stage == "E2":
        # Reasoning source: the ACF/PACF figure. Test whether a full day of history
        # beats 6 hours, and whether a second day adds anything.
        runs = [RunConfig(exp_id=f"E2-L{L}", model="gru", area=area, L=L,
                          hyperparams={"hidden": 64, "layers": 1},
                          note=f"input-length sweep L={L} ({L / 6:.0f}h), ACF-motivated")
                for L in (36, 144, 288)]

    elif args.stage == "E3":
        # Reasoning source: STL remainder variance scaled with level, and the marginal
        # distribution is right-skewed. Does log1p help?
        runs = [RunConfig(exp_id=f"E3-log1p{int(flag)}", model="gru", area=area,
                          L=args.L, use_log1p=flag,
                          hyperparams={"hidden": 64, "layers": 1},
                          note=f"preprocessing: log1p={flag}, from STL heteroscedasticity")
                for flag in (False, True)]

    elif args.stage == "E4":
        # Reasoning source: burstiness is near zero (B = -0.042), but RMSE is still
        # ~1.5x MAE and log1p did not reduce the largest errors. Huber caps their
        # gradient influence -- only if delta sits at the residual scale; the default
        # delta = 1.0 exceeds 99.9% of training residuals and would reduce to 0.5 * MSE.
        runs = [RunConfig(exp_id=f"E4-{loss}", model="gru", area=area, L=args.L,
                          use_log1p=args.log1p, loss=loss, huber_delta=args.huber_delta,
                          hyperparams={"hidden": 64, "layers": 1},
                          note=(f"loss={loss}" + (f", delta={args.huber_delta}"
                                                  if loss == "huber" else "")
                                + ", from RMSE/MAE ~1.5 and unreduced large errors"))
                for loss in ("mse", "huber")]

    elif args.stage == "E5":
        # Carry the winning preprocessing to the other two architectures. This tests
        # whether the choice is a property of the DATA or an artefact of the GRU.
        defaults = {"dlinear": {"kernel": 25}, "tcn": {"channels": 32, "dropout": 0.0},
                    "gru": {"hidden": 64, "layers": 1}}
        runs = [RunConfig(exp_id=f"E5-{name}", model=name, area=area, L=args.L,
                          use_log1p=args.log1p, loss=args.loss,
                          huber_delta=args.huber_delta,
                          hyperparams=defaults[name],
                          note="cross-model check of the Phase-1 preprocessing choice")
                for name in ("dlinear", "tcn", "gru")]

    print(f"--- stage {args.stage}: {len(runs)} run(s) ---")
    for run in runs:
        train_one(cfg, run)
    print(f"\nNow write your reasoning for {args.stage} into results/experiment_log.md "
          f"before running the next stage.")


if __name__ == "__main__":
    main()