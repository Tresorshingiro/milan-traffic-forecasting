from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Config:
    seed: int
    raw_dir: Path
    artifacts: Path
    results: Path
    n_squares: int
    slots_per_day: int
    n_days: int
    start_date: date
    tz: str
    train_days: tuple[int, int]
    val_days: tuple[int, int]
    test_days: tuple[int, int]
    n_top: int
    extra_areas: tuple[int, ...]
    batch_size: int
    max_epochs: int
    patience: int
    clip_norm: float

    @property
    def n_steps(self) -> int:
        return self.n_days * self.slots_per_day

    @property
    def matrix_path(self) -> Path:
        return self.artifacts / "internet.npy"

    @property
    def totals_path(self) -> Path:
        return self.artifacts / "square_totals.npy"

    @property
    def manifest_path(self) -> Path:
        return self.artifacts / "manifest.json"

    @property
    def memory_report_path(self) -> Path:
        return self.artifacts / "memory_report.json"

    @property
    def ledger_path(self) -> Path:
        return self.results / "experiments.jsonl"

    # Columns for day indices [d0, d1]
    def day_cols(self, d0: int, d1: int) -> slice:
        return slice(d0 * self.slots_per_day, d1 * self.slots_per_day)

    def split_cols(self, name: str) -> slice:
        days = {"train": self.train_days, "val": self.val_days, "test": self.test_days}[name]
        return self.day_cols(*days)

    def dates(self) -> list[date]:
        return [self.start_date + timedelta(days=i) for i in range (self.n_days)]


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path)
    if not path.is_absolute():
        path = _ROOT / path
    raw = yaml.safe_load(path.read_text())

    p, g, s, a, t = raw["paths"], raw["grid"], raw["splits"], raw["areas"], raw["training"]
    y, m, d = (int(v) for v in str(g["start_date"]).split("-"))

    return Config(
        seed=int(raw["seed"]),
        raw_dir= _ROOT / p["raw_dir"],
        artifacts= _ROOT / p["artifacts"],
        results= _ROOT / p["results"],
        n_squares=int(g["n_squares"]),
        slots_per_day=int(g["slots_per_day"]),
        n_days=int(g["n_days"]),
        start_date=date(y, m, d),
        tz=str(g["tz"]),
        train_days=tuple(s["train_days"]),
        val_days=tuple(s["val_days"]),
        test_days=tuple(s["test_days"]),
        n_top=int(a["n_top"]),
        extra_areas=tuple(a["extra"]),
        batch_size=int(t["batch_size"]),
        max_epochs=int(t["max_epochs"]),
        patience=int(t["patience"]),
        clip_norm=float(t["clip_norm"]),
    )

