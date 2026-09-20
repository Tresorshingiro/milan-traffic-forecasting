# Experiment log — Phase 1 manual ladder

Area: 5161 (highest-traffic square). All runs seed 42, on CPU (4 threads).
Raw records: `results/experiments.jsonl`.

Baselines to beat (from `results/tables/baselines.json`, test week Dec 16–22):
- persistence: MAE 92.80, MASE 0.267
- seasonal-naive: MAE 338.59, MASE 0.975

---

## E1 — Reference GRU (L=144, standardise, MSE, hidden 64, 1 layer, lr 1e-3)

**Result:** val MAE 102.37, test MAE 83.37, MASE 0.240, skill +0.760, 203.0 s
(12,929 parameters, 5,328 training windows, best epoch 27 of 38 run, 3.96 ms per
inference step)

**Observation:** The GRU beats both baselines on the test week. Against seasonal-naive
the gain is large (MAE 83.4 vs 338.6, about 75% lower), but that is the weaker baseline
here: on area 5161 persistence already reaches MASE 0.267, compared with seasonal-naive's
0.975. Measured against persistence, the GRU's margin is 9.4 MAE units, about 10%, or
0.027 in MASE. The skill score of +0.760 is therefore mostly the skill of predicting from
the previous slot at all, not of the GRU specifically. Persistence is the bar that matters.

Early stopping selected epoch 27 and stopped at epoch 38 (patience 10), well below the
100-epoch cap. Training converged rather than being cut short, so the result is not
limited by the epoch budget.

Validation MAE (102.4) is higher than test MAE (83.4). The model was selected on
validation, not test, so this gap is not overfitting to test. It suggests Dec 9–15 is a
harder week to forecast than Dec 16–22. The persistence baseline has not been scored on
the validation week, so this cannot yet be confirmed.

**Reasoning for the next change:** L=144 was chosen from the ACF's daily peak
(acf(144) = 0.854), but that choice is untested. The lag-1 autocorrelation is much higher
(acf(1) = 0.982), and persistence alone is already strong, so most of the predictive
signal may sit in the last few hours rather than a full day. E2 therefore sweeps
L ∈ {36, 144, 288} (6 h, 24 h, 48 h) to test two things: whether a shorter window gives up
accuracy, and whether a second day of history adds anything worth its cost. At 203 s per
run for L=144, training cost is now a real consideration alongside accuracy. The winner
is chosen on validation MAE, not test MAE, so the test week stays untouched by model
selection.

---

## E2 — Input length sweep, L in {36, 144, 288}

**Motivated by:** the ACF/PACF figure (`results/figures/05_acf_pacf.png`), where
acf(1) = 0.982 and acf(144) = 0.854.

| L | hours | val MAE | test MAE | MASE | train s | epochs (best / run) | s per epoch | inference ms/step |
|---|---|---|---|---|---|---|---|---|
| 36 | 6 | 105.35 | 83.34 | 0.240 | 47.6 | 18 / 29 | 1.6 | 0.94 |
| 144 | 24 | **102.37** | 83.37 | 0.240 | 200.6 | 27 / 38 | 5.3 | 3.87 |
| 288 | 48 | 104.80 | 83.96 | 0.242 | 258.6 | 12 / 23 | 11.2 | 7.63 |

**Result:** best L = 144 (lowest validation MAE)

**Observation:** Does more history help? Only a little, and not in proportion to its cost.
On validation, L=144 is best: L=36 is 2.9% worse (105.35) and L=288 is 2.4% worse
(104.80). On the test week the three lengths are effectively tied. Test MAE spans only
83.34–83.96 (0.7%), and MASE is 0.240–0.242 in every case. L=36 has the lowest test MAE,
but test results play no part in selection, and a 0.03 MAE difference is noise.

Cost scales roughly linearly with L, as expected for a recurrent model that steps through
every input position: training time per epoch goes from 1.6 s to 5.3 s to 11.2 s, and
inference from 0.94 to 3.87 to 7.63 ms per step. L=36 trains 4.2× faster than L=144 and
predicts 4.1× faster. It gives up 2.9% on validation and nothing measurable on test.

A second day of history (L=288) does not help. It was the worst on test and second on
validation. Its best epoch came earliest (12), which suggests that backpropagating
through 288 steps makes optimisation harder rather than adding usable signal. It also
trains on 252 fewer windows than L=36 (5,184 vs 5,436), because each window needs more
history before the first target. That is a small confound, about 5%.

The result matches the ACF: acf(1) = 0.982 is much higher than acf(144) = 0.854, so most
of the usable information sits in the most recent slots. The GRU gets little from
yesterday's same slot beyond what the last few hours already tell it. This is also why
persistence is such a strong baseline on this area.

**Reproducibility check:** E2-L144 has the same configuration as E1 and reproduced it
exactly: val MAE 102.371 and test MAE 83.368 in both runs, with the same best epoch (27)
and the same number of epochs run (38). Only wall-clock time differed (200.6 s vs 203.0 s).
Seeding therefore makes runs deterministic on this CPU setup.

**Caveat:** every run uses a single seed. The 2–3% validation gaps between lengths may be
within seed-to-seed variation, which this ladder does not measure. The ranking of L should
be read as weak evidence.

**Reasoning for the next change:** L=144 is carried forward. It follows the rule set
before E2 was run (select on validation MAE). It also keeps L within the TCN's receptive
field of 253 steps, which L=288 would exceed, so E5 can compare all three models at the
same input length. The cost trade-off is recorded for the final discussion: if training
time mattered more than a 3% validation gain, L=36 would be the better engineering choice.

With L fixed, the next lever is the input transform. Area 5161's standard deviation is
almost as large as its mean (CV 0.92, from `results/tables/profiles.csv`), and
test RMSE is about 1.5× test MAE (125.2 vs 83.4). That ratio indicates that a minority of
large errors, probably at the daily peaks, carry much of the squared error. E3 tests
whether log1p before standardising compresses those peaks enough to make them easier to
learn, while keeping every metric in original units.

---

## E3 — Preprocessing: standardise vs log1p + standardise

**Motivated by:** STL remainder variance scaling with level; right-skewed marginal.

| transform | val MAE | test MAE | MASE | test RMSE | test MAPE % | test sMAPE % | epochs (best / run) | train s |
|---|---|---|---|---|---|---|---|---|
| standardise | 102.37 | 83.37 | 0.240 | **125.18** | 8.62 | 8.34 | 27 / 38 | 194.9 |
| log1p + standardise | **101.06** | **82.73** | **0.238** | 125.55 | **8.15** | **7.89** | 45 / 56 | 280.1 |

**Result:** chosen = log1p + standardise (lowest validation MAE)

**Observation:** log1p helps, but only a little, and not in the way E2's reasoning
expected. Validation MAE falls by 1.3% (102.37 → 101.06) and test MAE by 0.8%
(83.37 → 82.73). The relative metrics improve more: test MAPE drops from 8.62% to 8.15%
and sMAPE from 8.34% to 7.89%, both about 5% lower.

The prediction from E2 was that log1p would compress the daily peaks and reduce the large
errors there. The data does not support that: test RMSE, which is dominated by the largest
errors, is essentially unchanged and slightly worse (125.18 → 125.55). The gain is instead
in proportional accuracy. Training with MSE on log-scale values penalises relative rather
than absolute error, so the model gives more weight to low-traffic hours (night and early
morning) and less to peaks. MAPE and sMAPE, which weight every hour by its relative error,
improve, while the peak-dominated RMSE does not.

log1p also slows convergence. The best epoch moves from 27 to 45, and training takes 44%
longer (280 s vs 195 s). Time per epoch is unchanged (about 5 s), so the extra cost comes
entirely from the extra epochs.

**Reproducibility check:** E3-log1p0 is the same configuration as E1 and E2-L144, and
reproduced them exactly for the third time: val MAE 102.371, test MAE 83.368, best epoch
27 of 38.

**Caveat:** a 1.3% validation gain from a single seed is small. It is consistent across
validation MAE, test MAE, MAPE, sMAPE and MASE, which makes it more credible than any one
number alone, but it is not a measured significance.

**Reasoning for the next change:** log1p is carried forward. It follows the rule set before
E3 was run (select on validation MAE), and it improves every metric except RMSE.

E4 was planned to test Huber against MSE, on the premise that area 5161 is bursty. The
measurement contradicts that premise. Its burstiness is B = −0.042 (from
`results/tables/profiles.csv`): slightly *more regular* than a Poisson process, not
heavy-tailed. The remaining case for Huber is weaker but real. Test RMSE is still 1.52×
test MAE, and E3 showed log1p does not reduce the largest errors. Capping their influence
on the gradient is a different mechanism that might.

Before running E4, there is a problem with the loss as implemented. `nn.HuberLoss()` uses
its default threshold δ = 1.0, applied to residuals in standardised units. On the training
split, under the log1p scaling (σ = 1.141), even the persistence forecast's absolute
residuals have a median of 0.084, a 90th percentile of 0.244, a 95th of 0.323 and a 99th
of 0.549. **Only 7 of 5,471 (0.13%) exceed 1.0.** The trained GRU's residuals are smaller
still. With δ = 1.0, Huber equals 0.5 × MSE on essentially every training sample. Adam is
nearly invariant to a constant scaling of the loss, so E4 as configured would measure
almost nothing. The only effect would be through gradient clipping at norm 1.0. A
meaningful comparison needs δ at the scale of the residuals, around the 90th–95th
percentile of training residuals (δ ≈ 0.25–0.3), so that only the largest errors are
capped. δ is taken from the training split only, so the test week plays no part in
setting it.

---

## E4 — Loss: MSE vs Huber

**Motivated by:** *not* burstiness, as originally planned. The measured burstiness is
B = −0.042 (from `results/tables/profiles.csv`), slightly more regular than a Poisson
process, so the heavy-tail premise does not hold for this area. The actual motivation is
from E3: test RMSE is still 1.52× test MAE, and log1p did not reduce the largest errors.
Huber caps the gradient influence of large residuals, which is a different mechanism from
log1p's compression of the input scale.

**Setup change before running:** δ = 0.3 in standardised log1p units, about the 95th
percentile of training-split residuals (see E3). The default δ = 1.0 exceeds 99.87% of
training residuals and would have made Huber equal to 0.5 × MSE. `huber_delta` was added
to `RunConfig` and is recorded in every ledger row. Both runs use L=144 and log1p.

| loss | val MAE | test MAE | MASE | test RMSE | test MAPE % | test sMAPE % | epochs (best / run) | train s |
|---|---|---|---|---|---|---|---|---|
| MSE | 101.06 | 82.73 | 0.238 | 125.55 | 8.15 | 7.89 | 45 / 56 | 292.6 |
| Huber (δ = 0.3) | **100.38** | **81.36** | **0.234** | **122.82** | **7.91** | **7.71** | 33 / 44 | 232.6 |

**Result:** chosen = Huber, δ = 0.3 (lowest validation MAE)

**Observation:** Huber improves every metric. The pattern of improvement is the reverse
of E3's. Validation MAE falls by 0.7% (101.06 → 100.38) and test MAE by 1.6%
(82.73 → 81.36). The largest relative gain is in RMSE, down 2.2% (125.55 → 122.82), the
metric that log1p left untouched. MAPE (−3.0%) and sMAPE (−2.3%) also improve. This
matches the mechanism: capping the gradient of the largest residuals stops a few
hard-to-predict slots from dominating the updates. The model then fits the bulk of the
series better, and the largest errors do not get worse.

Huber also converges faster: best epoch 33 instead of 45, and 20% less training time
(233 s vs 293 s). This fits the same picture: with the largest residuals no longer
dominating the gradient, the updates are more consistent from batch to batch.

The two preprocessing choices are complementary. log1p improved relative accuracy but not
RMSE. Huber on top of it improved RMSE as well.

**Reproducibility check:** E4-mse is the same configuration as E3-log1p1 and reproduced it
exactly for the fourth time: val MAE 101.056, test MAE 82.725, best epoch 45 of 56. (The
ledger records `huber_delta = 0.3` on the MSE row too; it has no effect when loss = MSE.)

**Cumulative progress over the ladder:** from E1 to E4, validation MAE has gone from
102.37 to 100.38 (−1.9%) and test MAE from 83.37 to 81.36 (−2.4%). Against persistence
(test MAE 92.80, MASE 0.267), the GRU is now 12.3% better, at MASE 0.234.

**Caveats:** the selection gain on validation is small (0.7%) and from a single seed. The
test gain is larger than the validation gain, but that cannot count toward the decision.
Only one δ was tried. It was set from training residuals before running, not tuned, so
the result is not the best Huber could do, but it is also not overfitted. Whether the
gain is concentrated at the daily peaks, or spread across the day, is not yet known. The
error-by-hour analysis in Task 13 will check this before the report draws that
conclusion.

**Reasoning for the next change:** Huber with δ = 0.3 is carried forward. It follows the
rule set before E4 was run (select on validation MAE), and it improves every metric while
training faster.

All three Phase-1 choices (L=144, log1p, Huber δ=0.3) were made on the GRU alone. E5
carries them to DLinear and TCN to test whether they transfer: are they properties of the
data, or of the GRU? L=144 is within the TCN's receptive field (253), so all three models
can run at the same input length. δ is defined in the same standardised log1p units for
every model, because the scaler is fitted once per area and does not depend on the model.
E5 is a check on the preprocessing choices, not the model comparison itself. That
comparison, including the pre-registered prediction that DLinear wins on area 5161, is
tested in Phase 2 across all three areas.

---

## E5 — Cross-model check of the Phase-1 preprocessing choice

All three models use L=144, log1p + standardise, Huber δ = 0.3, lr 1e-3, and default
hyperparameters (DLinear kernel 25; TCN 32 channels, dropout 0; GRU hidden 64, 1 layer).

| model | val MAE | test MAE | MASE | test RMSE | test MAPE % | params | epochs (best / run) | train s | inference ms/step |
|---|---|---|---|---|---|---|---|---|---|
| dlinear | 107.19 | 83.33 | 0.240 | 126.93 | 7.80 | 290 | 15 / 26 | 6.4 | 0.43 |
| tcn | **95.76** | **79.55** | **0.229** | **121.51** | **7.23** | 34,753 | 13 / 24 | 97.2 | 2.41 |
| gru | 100.38 | 81.36 | 0.234 | 122.82 | 7.91 | 12,929 | 33 / 44 | 229.2 | 3.86 |

Baselines for reference: persistence test MAE 92.80 (MASE 0.267), seasonal-naive 338.59
(MASE 0.975).

**Observation:** All three models beat both baselines under the Phase-1 settings. The TCN
is clearly best. Its validation MAE is 4.6% lower than the GRU's (95.76 vs 100.38) and
10.7% lower than DLinear's. It also leads on every test metric: MAE 79.55, which is 14.3%
better than persistence, RMSE 121.51 and MAPE 7.23%. The ranking is the same on
validation and test (TCN < GRU < DLinear). Model selection on validation therefore points
the same way as the held-out week.

Cost does not follow capacity. The TCN has 2.7× the GRU's parameters but trains 2.4×
faster (97 s vs 229 s) and predicts 1.6× faster (2.41 vs 3.86 ms per step). A convolution
processes all 144 input positions in parallel, while the GRU has to step through them in
sequence. The TCN also converged earliest (best epoch 13).

DLinear, with 290 parameters, trains in 6.4 s, 36× faster than the GRU, and still beats
persistence by 10.2% on test MAE (83.33). It is last on validation (107.19) and on RMSE
(126.93), but its test MAPE (7.80%) is slightly better than the GRU's (7.91%). A linear
model on log-scale inputs is multiplicative in original units, so it tracks
proportional changes well and misses most on large absolute deviations. Its weakness is
the large errors, not the typical error.

**Early signal against the pre-registered prediction:** `results/tables/prediction.json`
predicts DLinear as the winner on area 5161, the most predictable of the three areas. On
this area under Phase-1 settings, DLinear comes last on validation, and the
highest-capacity model wins. This is not yet a test of H1. The settings were tuned on the
GRU, and each model has only its default hyperparameters. The Phase 2 grid search tunes
each model on each area. The prediction file has not been edited.

**Reproducibility check:** E5-gru is the same configuration as E4-huber and reproduced it
exactly for the fifth time: val MAE 100.379, test MAE 81.364, best epoch 33 of 44.

**What E5 does and does not show:** The question in this stage's template was whether the
Phase-1 choices are a property of the data or specific to the GRU. As run, E5 cannot fully
answer it. Each model was trained only *with* the chosen settings, and never without
them, so there is no within-model comparison for DLinear or the TCN. E5 shows that the
settings work for all three models: every model trains stably and beats persistence. It
does not show that they are the best settings for DLinear or the TCN. A direct test would
rerun DLinear and the TCN under E1's settings (standardise, MSE). That costs about
2 minutes in total, and would show whether log1p + Huber helps them as it helped the GRU.

**Caveats:** single seed; default hyperparameters for each model; one area. The TCN's lead
on validation (4.6%) is larger than any single gain in E2–E4, which makes it the most
robust result in Phase 1. It is still one run per model.

**Conclusion carried into Phase 2:** L = 144, log1p = True, loss = Huber (δ = 0.3)

---

# Phase 2 — grid search and final runs

Areas: 5161 (busiest), 4159, 4556, the three target areas from `target_areas()`.
Fixed from Phase 1: L=144, log1p + standardise, Huber δ = 0.3. Seed 42, CPU, 4 threads.
Raw records: `E6-*` and `FINAL-*` rows in `results/experiments.jsonl`.

---

## E6 — Architecture grid, 20 configurations per area, 60 runs

**Grid:** DLinear kernel ∈ {25, 145}; TCN channels ∈ {16, 32} × dropout ∈ {0.0, 0.1};
GRU hidden ∈ {32, 64} × layers ∈ {1, 2}; each × lr ∈ {1e-3, 3e-4}. Selection on
validation MAE only. Total training time 4.1 h. The machine was CPU-throttled for the
first part of the search (see "Timing caveat" below), so E6 wall-clock times are not
comparable with each other; only the FINAL rows' timings are reported.

**Selected configurations (lowest validation MAE):**

| area | model | selected | val MAE | spread across grid |
|---|---|---|---|---|
| 5161 | dlinear | kernel 145, lr 1e-3 | 106.12 | 2.6% |
| 5161 | tcn | 32 ch, dropout 0.0, lr 1e-3 | **95.76** | 7.2% |
| 5161 | gru | hidden 64, 1 layer, lr 1e-3 | 100.38 | 3.7% |
| 4159 | dlinear | kernel 145, lr 1e-3 | 22.33 | 0.3% |
| 4159 | tcn | 32 ch, dropout 0.1, lr 1e-3 | **19.63** | 2.6% |
| 4159 | gru | hidden 64, 2 layers, lr 1e-3 | 20.52 | 2.1% |
| 4556 | dlinear | kernel 145, lr 3e-4 | 33.81 | 0.5% |
| 4556 | tcn | 32 ch, dropout 0.1, lr 1e-3 | **32.90** | 1.4% |
| 4556 | gru | hidden 64, 2 layers, lr 1e-3 | 33.02 | 4.7% |

"Spread" is (worst − best) / best validation MAE within one model's grid on one area.

**Observation:** On area 5161 the grid selected exactly the E5 configurations for the
TCN (32 channels, no dropout) and the GRU (hidden 64, 1 layer). Only DLinear changed,
from kernel 25 to 145, a 1.0% validation gain. The Phase-1 defaults were therefore
already the best in the grid on the area they were tuned on.

Three patterns hold across areas:

1. **DLinear prefers kernel 145 everywhere.** A moving average spanning a full day
   separates the daily cycle from the slower level, which suits a daily-periodic series.
   The effect is tiny (spread 0.3–2.6%): a 290-parameter linear model has little room to
   vary.
2. **The larger TCN (32 channels) wins on every area**, and dropout 0.1 helps on the two
   smaller areas (4159, 4556) but not on 5161. The two smaller areas are noisier in
   relative terms (lower level, same absolute noise), which is where regularisation
   would be expected to help.
3. **The GRU moves to 2 layers on the smaller areas** but stays at 1 layer on 5161.
   Two-layer GRUs are also the slowest configurations by a wide margin.

The learning rate matters little. Averaged over each model's grid, lr 1e-3 and 3e-4
differ by at most 2% in validation MAE (TCN on 5161), and in most cells by under 0.5%.
lr 3e-4 mainly costs more epochs. 8 of 9 selected configurations use lr 1e-3.

Model choice matters more than any hyperparameter. The grid spread within a model
(0.3–7.2%) is of the same order as the gap between models, so the architecture
comparison below is sensitive to single-seed noise.

**Test-set discipline check:** the best *test* result in the grid is not always the
selected one. On 5161, `E6-tcn-02` (16 channels, dropout 0.1) has test MAE 77.71, better
than the selected TCN's 79.55, but it ranks fourth on validation and was not chosen.
Selecting on test would have inflated the TCN's result by 2.3%.

**Timing caveat:** for roughly the first hour of the grid the CPU ran at about 1.2 GHz
(of 3.5 GHz) under the `balanced` power profile with other applications open; it was
then switched to `performance` (about 2.3 GHz). This does not affect accuracy — every
run is deterministic, and all nine FINAL re-trains reproduced their grid rows'
validation MAE exactly — but it makes E6 wall-clock times unreliable.

---

## E7 — Final runs: best configuration per (model, area), scored on the test week

Each selected configuration was re-trained once with inference timing and saved
predictions (`FINAL-*` rows). All nine reproduced their E6 validation MAE exactly.
Tables: `results/tables/results_area_<id>.csv`, `timing.csv`, `hypothesis_test.json`.

**Test week (Dec 16–22) results:**

| area | model | MAE | RMSE | MAPE % | sMAPE % | MASE | params |
|---|---|---|---|---|---|---|---|
| 5161 | persistence | 92.80 | 134.88 | 9.19 | 9.10 | 0.267 | 0 |
| 5161 | seasonal-naive | 338.59 | 619.04 | 25.94 | 22.83 | 0.975 | 0 |
| 5161 | dlinear | 83.21 | 127.04 | 7.70 | 7.64 | 0.240 | 290 |
| 5161 | **tcn** | **79.55** | **121.51** | **7.23** | **7.14** | **0.229** | 34,753 |
| 5161 | gru | 81.36 | 122.82 | 7.91 | 7.71 | 0.234 | 12,929 |
| 4159 | persistence | 15.95 | 21.54 | 6.98 | 6.94 | 0.195 | 0 |
| 4159 | seasonal-naive | 51.19 | 84.64 | 21.80 | 20.49 | 0.626 | 0 |
| 4159 | dlinear | 14.28 | 19.54 | 6.14 | 6.14 | 0.175 | 290 |
| 4159 | **tcn** | **13.57** | **18.75** | **5.72** | **5.72** | **0.166** | 34,753 |
| 4159 | gru | 14.07 | 19.40 | 6.13 | 6.07 | 0.172 | 37,889 |
| 4556 | persistence | 28.86 | 39.62 | 6.60 | 6.55 | 0.257 | 0 |
| 4556 | seasonal-naive | 76.34 | 108.35 | 17.46 | 15.87 | 0.680 | 0 |
| 4556 | **dlinear** | **25.61** | **34.75** | **5.79** | **5.76** | **0.228** | 290 |
| 4556 | tcn | 25.91 | 34.82 | 5.95 | 5.84 | 0.231 | 34,753 |
| 4556 | gru | 26.76 | 36.10 | 6.13 | 6.00 | 0.238 | 37,889 |

No MAPE targets were excluded (every test value is above the 1e-3 threshold).

**Cost (FINAL runs, performance power profile):**

| area | model | train s | inference ms/step | epochs (best / run) |
|---|---|---|---|---|
| 5161 | dlinear | 8.0 | 0.16 | 22 / 33 |
| 5161 | tcn | 132.3 | 3.69 | 13 / 24 |
| 5161 | gru | 305.0 | 5.32 | 33 / 44 |
| 4159 | dlinear | 7.8 | 0.17 | 20 / 31 |
| 4159 | tcn | 240.7 | 3.54 | 25 / 36 |
| 4159 | gru | 506.9 | 10.56 | 25 / 36 |
| 4556 | dlinear | 18.8 | 0.16 | 70 / 81 |
| 4556 | tcn | 236.8 | 3.49 | 26 / 37 |
| 4556 | gru | 748.5 | 10.59 | 42 / 53 |

**Observation — every model beats both baselines on every area.** Against persistence,
the demanding baseline here, test MAE falls by 10.3–14.3% on 5161, 10.5–14.9% on 4159
and 7.3–11.3% on 4556. Against seasonal-naive the gains are 65–77%, but seasonal-naive
is weak at a 10-minute horizon: on 5161 its MASE is 0.975, barely better than itself
in-sample.

The three models are close. Within each area, the best and worst model differ by
4.6% (5161), 5.2% (4159) and 4.5% (4556) in test MAE. With one seed, that is weak
evidence for any ranking.

**Validation and test disagree on area 4556.** On validation the order is
TCN < GRU < DLinear on all three areas. On the test week it is the same for 5161 and
4159, but on 4556 it flips to DLinear < TCN < GRU. DLinear's test lead there is 1.2%
over the TCN. A procedure that selects on validation would pick the TCN on every area.

**Cost is where the models separate.** DLinear reaches within 5.2% of the best test
MAE on every area (and wins one) with 290 parameters, 0.8% of the TCN's. It trains in
8–19 s against 2–12 min, and predicts 21–66× faster per step (0.16 ms against
3.5–10.6 ms). The TCN is again faster than the GRU at similar or larger size, because
it convolves over all 144 inputs in parallel.

---

## Hypothesis tests

**H1 — not supported.** The pre-registered prediction (`prediction.json`, committed
before any model was trained) was DLinear on 5161 and 4159, and TCN/GRU on 4556. The
test-week winners by MASE were TCN on 5161, TCN on 4159, and DLinear on 4556: the
prediction failed on all three areas, and on 4556 the outcome was the exact reverse.
Using the validation ranking instead (TCN everywhere), the prediction holds only on 4556,
1 of 3.

Two caveats limit what this negative result means. First, the prediction rule split the
three areas at the median predictability score, so it assigned DLinear to two of three
areas by construction, whatever the profiles said. Second, the model gaps (≤ 5%) are of
the same order as single-seed and grid variation, so "winner" is a fragile label here.
What the data does support is weaker but more interesting: capacity did not buy much
anywhere. The most "predictable" area by the profile score (5161) is also where the
TCN's advantage is largest, the opposite of what H1 expected.

**H2 — supported.** Within one area MAE and MASE always rank models identically, because
MASE divides by a per-area constant. Across areas they disagree. Ranked easiest to
hardest, raw MAE puts the areas in the order 4159, 4556, 5161 for every model — the same
as their traffic volume. By MASE, the TCN and the GRU rank 5161 (0.229, 0.234) as easier
than 4556 (0.231, 0.238). Raw MAE would call the busiest area the hardest for these
models, when relative to its own seasonal-naive benchmark it is not.

---

## Failure analysis (`results/tables/failure_analysis.json`, figures 09–10)

**1. Phase lag — the clearest failure, and it affects every model.** Shifting each
model's forecast one step earlier reduces its test MAE substantially:

| area | model | MAE as issued | MAE shifted by 1 | reduction |
|---|---|---|---|---|
| 5161 | dlinear | 83.36 | 59.74 | 28% |
| 5161 | tcn | 79.70 | 69.94 | 12% |
| 5161 | gru | 81.53 | 58.67 | 28% |
| 4159 | dlinear | 14.30 | 9.07 | 37% |
| 4159 | tcn | 13.58 | 10.59 | 22% |
| 4159 | gru | 14.08 | 9.66 | 31% |
| 4556 | dlinear | 25.55 | 16.27 | 36% |
| 4556 | tcn | 25.81 | 19.24 | 25% |
| 4556 | gru | 26.70 | 19.34 | 28% |

(MAE here is computed on the 1,005 points common to all shifts, so it differs slightly
from the table above.) Each forecast partly tracks the previous observed value rather
than anticipating the next one: part of what the models learned is persistence. The
TCN is the least lagged model on every area, which fits its leading accuracy. On 5161
the residuals also keep lag-1 autocorrelation of about +0.3 for all three models, so
consecutive errors are not independent.

**2. Error by hour of day.** Errors are small overnight (MAE about 10–50 on 5161) and
concentrate between 09:00 and 20:00, peaking near 13:00–15:00 at up to about 285. That is where
traffic is highest and changes fastest. In relative terms the pattern is the reverse,
which is why MAPE is sensitive to the night-time regime.

**3. Day-to-day drift — not the holiday effect the plan expected.** On 5161, error is
highest on Sat Dec 21 and Sun Dec 22. But those days' peaks (5,238 and 5,496) are within
the range of training weekends (up to 8,044): 5161 simply peaks at weekends, and larger
values bring larger absolute errors. On 4159 and 4556 the drift runs the other way:
traffic falls into December. On 4159, weekday peaks drop from about 745 in the
validation week to about 520 in the test week, against up to 941 in training. This lower
level is a genuine distribution shift that the chronological split exposes.

This also answers the question left open in E1 (why validation MAE exceeds test MAE).
On every area, the ratio of test to validation MAE tracks the ratio of traffic levels
between the two weeks: on 4159, weekday peaks fall to about 0.70 of their validation
level and the TCN's MAE to 0.69 of it; on 4556, about 0.8 and 0.79. Absolute error
scales with traffic, and the test week is simply quieter than the validation week.

**4. Worst single errors.** On 5161, Dec 17 at 15:30–16:00 (errors 565–616), a sharp
afternoon swing. On 4556, Dec 17 at 00:40 (error 294): a one-slot spike from about 415
to 718 and straight back. No one-step model can anticipate an isolated spike, and a
persistence-like model is then wrong twice — at the spike and on the slot after it.

---

## Conclusion of Phase 2

All three architectures beat both baselines on all three areas, by 7–15% over
persistence. The TCN is best on validation everywhere and on test in two of three areas,
but the margins are within single-seed noise. DLinear matches the deep models to within
5.2% at a small fraction of their cost. H1 is not supported; H2 is. The main shared
weakness is phase lag: every model is partly persistence.

**Open items:** multiple seeds for the nine final configurations (turns the ≤ 5% gaps
into a testable claim); DLinear and TCN under E1's settings (the missing E5 control).
