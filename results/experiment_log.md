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
