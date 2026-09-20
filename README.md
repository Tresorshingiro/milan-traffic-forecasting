# Forecasting Mobile Network Traffic in Milan

Comparative study of three sequential forecasting models — **DLinear**, a **TCN**, and a
**GRU** — for one-step-ahead (10-minute) Internet traffic prediction on the Telecom Italia
Milan grid, framed around a single question: *does a measurable forecastability profile of
a geographical area predict which model class will win there?*

## Research questions

- **H1** — Areas with strong deterministic daily periodicity are best served by a
  low-capacity linear forecaster; high-entropy, bursty areas are where nonlinear capacity
  pays for itself.
- **H2** — Raw MAE/RMSE cannot support cross-area comparison when areas differ in volume
  by an order of magnitude; MASE, referenced to each area's own seasonal-naive error, can.

The H1 prediction is **pre-registered** in `results/tables/prediction.json`, committed in
`90cde5b` before any forecasting model was trained.

## Data

Telecom Italia Big Data Challenge, Milan (Barlacchi et al., 2015). 62 daily files,
2013-11-01 to 2014-01-01, 20.8 GB, tab-separated:

`squareId | timeInterval(ms) | countryCode | smsIn | smsOut | callIn | callOut | internet`

Download the `sms-call-internet-mi-*.txt` files into `milan_dataset/`
(https://doi.org/10.7910/DVN/EGZHFV). They are not in this repository.

**A blank field is a structural zero, not missing data.** The file is a sparse
`(square × time × countryCode)` cube, so a blank `internet` value means no traffic was
attributed to that country in that cell during that slot. The correct reduction sums
across all country codes treating blanks as zero; dropping null rows would discard
real traffic.

Splits are chronological: train Nov 1 – Dec 8, validation Dec 9 – 15, test Dec 16 – 22
(fixed by the brief). Dec 23 – Jan 1 is unused.

## Setup

Requires Python 3.12 and `mawk` (`sudo apt install mawk`).

```bash
make setup
```

A virtual environment is mandatory, not stylistic: a system Python with `numpy` 2.x
alongside `scipy`/`matplotlib` built for 1.x raises
`ImportError: numpy.core.multiarray failed to import`. PyTorch is installed CPU-only
from the PyTorch index, which is why it is not in `requirements.txt`.

## Reproducing the results

```bash
make etl        # 20.8 GB -> 357 MB float32 memmap        (~7 min)
make memory     # naive-vs-streaming memory evidence
make eda        # exploratory figures 01-06
make profile    # forecastability profiles + pre-registered prediction
```

Phase 1 is **deliberately interactive** — the reasoning between stages is part of the
method, so the stages are not chained:

```bash
PYTHONPATH=src .venv/bin/python scripts/experiment_ladder.py E1
# record the result and your reasoning in results/experiment_log.md, then continue
PYTHONPATH=src .venv/bin/python scripts/experiment_ladder.py E2
# ... E3/E4/E5, passing forward the winning --L, --log1p, --loss
```

Then:

```bash
make grid       # 60 runs, 20 per area (~3 h on 4 CPU threads); resumes if interrupted
make final      # re-trains the 9 selected configs; 3 tables, 9 plots, hypothesis verdict
make failure    # phase-lag, error-by-hour, and error-by-day diagnostics
```

`make grid` uses the Phase-1 choices by default (`L=144`, log1p, Huber with δ = 0.3 in
standardised units); override with e.g. `make grid L=36 LOSS=mse LOG1P= `.

Every run appends one JSON line to `results/experiments.jsonl` with its full
configuration, metrics, and timings. All selection uses validation MAE; the test week is
scored once per final configuration.

## Memory management

| | Naive | This pipeline |
|---|---|---|
| Approach | `pd.read_csv` of one whole day-file, all 8 columns, float64 | streaming `mawk` reduction into a `float32` memmap |
| Input | one 343 MB file (Dec 16) | all 62 files, 20.8 GB |
| Output | 340.5 MB DataFrame for **one day** | 357 MB (`10000 × 8928` float32) for **all 62 days** |
| Reduction | — | **58×** (raw bytes / artifact bytes) |
| Peak RSS | 716 MB for one day | 419 MB for all 62 days |
| Time | ~5 s for one file | 7.2 s median per file, 438 s total |

Measured figures are in `artifacts/memory_report.json` and, per day, in
`artifacts/manifest.json`. The ETL is idempotent via the manifest, so an interrupted run
resumes instead of restarting.

## Layout

```
src/milan/          library: etl, data, analysis, models, train, evaluate, plotting
scripts/            thin CLI entry points
tests/              pytest suite (`make test`, or `make test-fast` to skip dataset tests)
artifacts/          memmap, manifest, memory report          (gitignored)
results/            experiments.jsonl, experiment_log.md, tables/, figures/, predictions/
report/             outline for the written report
```

## Testing

```bash
make test-fast   # unit tests only
make test        # includes tests that read the real dataset and train small models
```

The suite covers the silent-failure cases: window/target leakage, scalers fitted outside
the training split, the awk reducer's arithmetic (against independently computed golden
values), TCN receptive-field coverage, the complexity measures (against white noise, a
random walk, a pure tone, and a monotone ramp), and the failure diagnostics (against
forecasts with a known one-step lag).

## Report, code, and video

- Report: (https://docs.google.com/document/d/1djuS5BfEGswUqrNFpP9VkG1wCiaibfBnKtV-DlDRxHA/edit?usp=sharing)
- Video: (https://drive.google.com/file/d/1RYTeZHuzO_rOjU73nLzzEZoQEyev8UU0/view?usp=sharing)
