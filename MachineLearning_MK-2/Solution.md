# Solution design — MachineLearning_MK-2

A PyTorch 1D-CNN regressor that predicts `speed` (km/h) for one road-segment
measurement from the `SegmentSpeed` covariates named in the task spec: `statusEnum`,
`dayNr`, `daysUntilHoliday`, `holidayType`, `minutesSincDaybreak`, `monthOfYear`,
`holidayNum`.

## Why a CNN for row-wise tabular data

The task explicitly asks to "base the model on a CNN." That's worth being upfront
about: a 1D convolution's core advantage — a small kernel sliding across a signal so
the same learned filter recognizes a pattern wherever it appears — is a natural fit
for a *sequence* (time, or ordered spatial position). Here there's no such axis: the 7
features aren't samples of one continuous signal, they're heterogeneous, unordered
covariates (a status code, three cyclical calendar quantities, a holiday countdown,
two holiday categoricals).

What's implemented (`model.py::TabularCNNRegressor`) is the standard way to still use
a CNN on tabular data: concatenate all engineered features (7 continuous + 3 learned
categorical embeddings) into one fixed-order vector, treat it as a 1-channel,
length-17 signal, and run `Conv1d` layers over it before pooling to a fixed-size
vector for the regression head. The convolutions end up doing local feature-mixing
between whichever engineered columns happen to sit next to each other in the
concatenation order (features.py's `continuous_features` orders them
`[day_sin, day_cos, month_sin, month_cos, min_sin, min_cos, days_scaled]`, followed by
the three embedding blocks) — a real computation, but not one with the translation-
invariance rationale that makes CNNs the obvious choice for images or true time
series. `AdaptiveAvgPool1d(1)` after the conv stack makes the model tolerant of the
resulting vector length changing if the embedding dims are re-tuned (see
`--status-embed-dim` etc. in Operations.md), instead of hard-coding a flattened size.
An `nn.Linear`-only MLP over the same 17 features would very likely match or beat this
CNN's accuracy at lower parameter count and compute cost, precisely because there's no
spatial structure for the convolution to exploit — see "Suggested improvements" below.

## What's *not* in this model, on purpose

The task's feature list for `SegmentSpeed` does not include `sectionId`. That is a
significant limitation worth stating plainly: different road segments have very
different free-flow speeds (a residential street vs. a motorway), and without a
segment identity or any spatial signal, the model can only learn one global
speed-vs-time-and-status curve, not "this specific segment's" curve. `MK-1`'s
segment-embedding forecaster does not have this limitation. If per-segment accuracy
matters for the downstream congestion model, adding `sectionId` as an embedding
feature (exactly as `MK-1` does) is the highest-leverage change available — see
"Suggested improvements."

## Data contract

Same `SegmentSpeed` JSON array as `MK-1` (see `REST-opperations.md` in the repo root
for full field semantics; `holidayType` is the task's name for the API's
`holidayNr`). Only the fields below are used — `sectionId` and `measureTime`, if
present in the export, are read but ignored:

| Field | Type | Notes |
|---|---|---|
| `statusEnum` | int 0–3 | `freeflow=0, heavy=1, congested=2, impossible=3`; embedded |
| `speed` | float | km/h; the regression target |
| `dayNr` | int 0–6 | ISO weekday, Monday=0; cyclically encoded |
| `monthOfYear` | int 1–12 | cyclically encoded |
| `minutesSincDaybreak` | int 0–1439 | minutes since 06:00, wraps past midnight; cyclically encoded |
| `daysUntilHoliday` | int | standardized (zero mean, unit variance) |
| `holidayType` | int 0–2 | regular/eve/holiday; embedded |
| `holidayNum` | int 0–14 | which named holiday is next; embedded (0 unused/reserved) |

Rows with a null value in any required field are dropped up front
(`dataset.py::load_dataframe`).

## Feature engineering

- **Cyclical features** (`features.py::cyclical_encode`): `dayNr`, `monthOfYear`, and
  `minutesSincDaybreak` are each encoded as `(sin, cos)` pairs, so e.g. minute 1439 and
  minute 0 are seen as adjacent rather than maximally distant.
- **Scaled continuous feature**: `daysUntilHoliday` is standardized with
  `sklearn.preprocessing.StandardScaler`, **fit on the training split only**
  (`dataset.py::fit_scalers`) to avoid test-set leakage into the normalization
  statistics. This is the "normalize input data" requirement from the task.
- **Categorical embeddings**: `statusEnum` (4), `holidayType` (3), `holidayNum` (15, 0
  reserved) each get a small learned `nn.Embedding` table
  (`model.py::TabularCNNRegressor`), sized by `--status-embed-dim`,
  `--holiday-type-embed-dim`, `--holiday-num-embed-dim`.

## Target transform: log vs. raw speed

The task asks to "try out if target variable should be transformed with log or not."
`dataset.py::TargetTransform` supports both (`--log-target`), applying `log1p` before
standardizing (both fit on the training split only) when enabled, and inverting
(`expm1` then un-standardize) for every reported metric so results are always in km/h.

**Measured on the real `raw-trafic-data.json` export** (107,953 rows, 100 sections,
~3 days of collection as of 2026-09-15 — see caveats below), same architecture and
seed, 15 epochs each:

| Target | Best val MAE | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| raw `speed` | 8.91 km/h | 8.92 km/h | 11.05 km/h | 0.396 |
| `log1p(speed)` | 9.12 km/h | 9.11 km/h | 11.10 km/h | 0.390 |

Raw speed wins, narrowly but consistently across train/val/test. This is the opposite
of the usual advice for a strictly-positive, right-skewed target (where `log1p`
compresses a long right tail and often helps): this collector's `speed` is
**left-skewed** (skew ≈ −0.97 on the current export) because it's *upper-bounded* near
each segment's free-flow speed and only has room to fall, not rise, when congested.
`log1p` compresses the low end (which is where the interesting, hard-to-predict
congestion variation lives) and stretches the already-dense high end, making the
learning problem slightly harder rather than easier. `--log-target` defaults to
**off** in `train.py`/`tune.py` accordingly. Re-run the comparison
(`Operations.md`'s "Comparing raw vs. log target" recipe) if the collector's speed
distribution changes materially (e.g. once slower urban segments are added — see
caveats below).

## Model (`model.py::TabularCNNRegressor`)

```
statusEnum ──► embedding (dim=status_embed_dim)      ┐
holidayType ──► embedding (dim=holiday_type_embed_dim)├─ concat ──┐
holidayNum ──► embedding (dim=holiday_num_embed_dim)  ┘           │
                                                                    ▼
dayNr, monthOfYear, minutesSincDaybreak (sin/cos ×3)   concat ──► (B, 1, L) 1-channel signal
daysUntilHoliday (scaled)              ──────────────┘             │
                                                                    ▼
                                                        Conv1d → BatchNorm1d → ReLU → Dropout
                                                              (× len(--conv-channels))
                                                                    │
                                                                    ▼
                                                          AdaptiveAvgPool1d(1)
                                                                    │
                                                                    ▼
                                                     Linear → ReLU → Dropout → Linear
                                                                    │
                                                                    ▼
                                                        speed (km/h, after inverse
                                                           target transform)
```

- `--conv-channels` (default `32,64`) sets the number and width of `Conv1d` layers;
  `--kernel-size` (default 3) and `--dropout` (default 0.1) are shared across them.
- `BatchNorm1d` after every conv stabilizes training given the small (17-wide) input;
  without it, this net was noticeably more sensitive to `--lr`.
- Loss is plain `MSELoss` on the *scaled* target; `run_epoch` (`train.py`) also
  computes MAE in km/h every epoch (inverse-transforming predictions) so training logs
  are directly interpretable without a separate evaluation pass.

## Train/test split

Per the task spec: `sklearn.model_selection.train_test_split(row_indices,
test_size=0.1, random_state=4711, shuffle=True)` (`dataset.py::split_train_test`),
applied to plain row indices since there is no windowing or per-segment grouping to
respect (contrast with `MK-1`, where windows had to stay within one segment). A
further slice of the 90% training partition (`--val-fraction`, default 0.1 of *that*
partition, same seed) is held out from gradient updates for model
selection/early-stopping only; feature and target scalers are fit on the remaining
train-only rows (excluding both the validation slice and the test set), and the
mandated 10% test set is touched only by the final `evaluate.py` run.

Because rows are i.i.d. samples with no sequence structure, this random shuffle has
none of `MK-1`'s "near-duplicate window on both sides of the split" leakage concern —
each row is an independent measurement.

## Hyperparameter tuning (`tune.py`)

Random search (not grid — the space is large enough and each trial is inexpensive)
over `--conv-channels`, `--kernel-size`, `--dropout`, `--lr`, `--batch-size`, the three
embedding dims, and `--log-target`, seeded by `--search-seed` (default 4711, same
constant as the data split, but a logically separate seed — see Operations.md). Each
trial trains for a short `--trial-epochs` budget and is ranked by validation MAE
(km/h); the winning configuration is retrained for the full `--final-epochs` budget
and checkpointed. `tune.py` calls `train.py::run_training` directly rather than
shelling out, so tuning and standalone training always run identical code.

**Search results on the real export** (10 trials, 5 epochs each, final retrain 20
epochs; full trial table in `tuning/tuning_results.csv` after running `tune.py`):

| Trial | conv_channels | kernel | dropout | lr | batch | status/holidayType/holidayNum embed dims | log_target | val MAE (5-epoch trial) |
|---|---|---|---|---|---|---|---|---|
| 4 (winner) | 32,64 | 3 | 0.0 | 3e-3 | 256 | 2/2/2 | True | **8.76 km/h** |
| 1 | 32,64 | 5 | 0.0 | 1e-3 | 512 | 8/2/4 | True | 8.87 km/h |
| 9 | 64,64 | 5 | 0.0 | 1e-3 | 256 | 4/2/4 | True | 9.01 km/h |
| 6 | 32,64 | 3 | 0.1 | 3e-4 | 256 | 4/2/4 | False | 9.01 km/h |
| 3, 5 | — | — | — | — | — | — | — | 9.08 km/h |
| 10 (worst) | 16,32 | 3 | 0.3 | 3e-4 | 1024 | 4/2/8 | False | 9.80 km/h |

Retraining trial 4's config for the full 20 epochs, then evaluating on the held-out
test set: **MAE 8.80 km/h, RMSE 11.38 km/h, MAPE 18.66%, R² 0.359** — a lower MAE than
the untuned raw-target baseline above (8.92 km/h) but a *higher* RMSE and lower R²,
i.e. the tuned model trades away some consistency (a few larger errors) for a better
average. Whether that trade is worth it depends on the downstream use case (MAE-
optimal vs. penalizing large misses more).

**A caveat about trusting this search's `log_target=True` pick**: the winning trial's
best validation MAE (8.76 km/h) was reached at epoch 2 of its 5-epoch budget, then
never beaten again during the full 20-epoch retrain (which fluctuated 8.97–10.7 km/h
across epochs, only dipping back near 8.76 twice, at epochs 2 and 16 — see the run log
in `checkpoints/history.json` after reproducing it). With a validation slice of under
10,000 rows and only 5 epochs per trial, a single lucky early-epoch checkpoint can
outscore genuinely better configs that hadn't converged yet — random search here is
noisier than the trial count suggests. The controlled, matched-hyperparameter
log-vs-raw comparison earlier in this section (15 full epochs each, only the target
transform varied) is the more trustworthy signal on the log-target question;
`--log-target` still defaults to off in `train.py`/`tune.py` for that reason, even
though this particular search nominally preferred it. See "Suggested improvements"
for how to make the search itself less noisy.

## Evaluation (`evaluate.py`)

Loads a checkpoint, rebuilds the *same* test split deterministically (same seed, read
back from the checkpoint's saved training args — the split itself isn't persisted),
and reports MAE, RMSE, MAPE, and R² in km/h, plus a predicted-vs-true scatter plot and
a residual histogram (`checkpoints/eval/prediction_plots.png`).

## Suggested improvements

**Modeling**
- **Add `sectionId` as an embedding feature** (as `MK-1` already does) — the single
  highest-leverage change. Without it the model is fundamentally capped at predicting
  the *average* speed-vs-covariates curve across all segments; a motorway and a
  residential street currently get the same prediction for the same time-of-day/status
  combination. This is very likely the dominant source of the current ~9 km/h MAE /
  R²≈0.4, more than any architecture or hyperparameter change below.
- **Compare against a plain MLP baseline.** Given the "why a CNN" discussion above, an
  `nn.Linear`-stack over the same 17-dim feature vector is the natural ablation to run
  once real infrastructure is available — if it matches or beats the CNN (plausible,
  given no spatial structure in the input), that's a simpler, cheaper model to deploy
  and maintain; if the CNN genuinely wins, that's useful evidence the local
  feature-mixing is doing something.
- **`statusEnum` is a very strong, almost circular predictor of `speed`** (both are
  derived from the same underlying measurement — see `SpeedOfSectionMLData.java`'s
  status thresholds). It's included because the task lists it explicitly, but a model
  meant to run at inference time on genuinely *future* covariates (the actual
  congestion-prediction use case) won't have next-minute's `statusEnum` available any
  more than it has next-minute's `speed` — that's circular. Clarify with the
  downstream consumer whether `statusEnum` is really available at prediction time; if
  not, drop it and expect MAE to rise, but the resulting model will reflect the real
  deployment task instead of a much easier proxy problem.
- Add feature interactions the current concatenation-then-conv design may not surface
  well: e.g. an explicit `is_weekend` bit, or a joint day×minute cyclical feature (rush
  hour shape differs materially between weekday and weekend, which today the model can
  only infer by combining two separate cyclical encodings through the conv/FC layers).

**Data / evaluation caveats**
- **The current `raw-trafic-data.json` export is not yet representative.** As collected
  so far (100 sections, ~3 calendar days in September 2026), `holidayType` is always 0,
  `holidayNum` is always 10, and `monthOfYear` is always 9 — three of the seven task
  features carry **zero information** in the current data (see the value-count dump
  this analysis was based on). The model can't learn anything about holidays or
  seasonality until the collector has run across an actual holiday and multiple
  months; today's reported metrics measure only the time-of-day/weekday/status
  relationship. Re-run tuning and evaluation once more calendar diversity has
  accumulated — the current numbers are a pipeline-correctness check, not a
  production-readiness measure.
- Because of the above, the log-vs-raw-target result and the tuned hyperparameters in
  this document should both be treated as provisional; re-run the comparisons in
  Operations.md periodically as the export grows.
- A block-in-time split (e.g. most recent N days held out) would be a better measure
  of genuine forecasting skill once enough calendar range exists to make that
  meaningful; the random 90/10 split (as specified in the task) is fine for now since
  rows are i.i.d. with no leakage risk, but it never tests generalization to an unseen
  time period the way a temporal holdout would.

**Engineering**
- `tune.py`'s ranking uses each trial's *best* validation MAE across its short epoch
  budget, which — as the tuning results above show — lets an early, noisy minimum win
  over configs that were still converging. Ranking by an average of the last few
  epochs (or a longer, fixed `--trial-epochs`) would make the search less susceptible
  to picking a lucky checkpoint over a genuinely better configuration.
- `tune.py`'s random search has no early termination for clearly-bad trials (e.g. a
  Hyperband/ASHA-style scheduler) — with only ~10s/epoch on CPU this doesn't matter
  yet, but would on a much larger export or deeper search space.
- Learning-rate scheduling (`ReduceLROnPlateau`) and early stopping are not
  implemented; `train.py` runs a fixed `--epochs` and keeps the best-val checkpoint,
  but never stops early or decays the LR. Val MAE plateaus by epoch ~10 on the current
  data (see the training logs in `checkpoints/history.json`), so this mainly wastes
  compute rather than hurting accuracy today.
