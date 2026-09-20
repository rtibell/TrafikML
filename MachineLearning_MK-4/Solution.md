# Solution design — MachineLearning_MK-4

A PyTorch multilayer perceptron (MLP) that **regresses `speed`** (km/h) for one
road-segment measurement, from the revised `SegmentSpeed` feature set: `dayNr`,
`daysUntilHoliday`, `minutesSincDaybreak`, `monthOfYear`, and 17 upstream one-hot
fields (`holiday_type_*`, `*_status`, `holiday_is_*`).

This is the fourth iteration of the modelling approach in this repository:

| Version | Task | Architecture | Categorical inputs |
|---|---|---|---|
| `MK-1` | regress `speed` | LSTM/GRU sequence model, `sectionId` embedded | per-segment minute-by-minute time series |
| `MK-2` | regress `speed` | 1D-CNN over a concatenated feature vector | `statusEnum`/`holidayType`/`holidayNum` embedded |
| `MK-3` | classify `statusEnum` | MLP, softmax output | `holidayType`/`holidayNum` embedded |
| **`MK-4` (this version)** | **regress `speed`** | **MLP** | **none — upstream one-hot fields fed directly** |

## What changed from `MK-3`, and why

Two independent things changed at once, both driven by the task's revised
`SegmentSpeed` structure:

1. **The upstream API stopped sending small-integer categoricals and started sending
   one-hot fields instead.** `holidayType` (0/1/2) became `holiday_type_regular_day` /
   `holiday_type_eve` / `holiday_type_holiday`; `holidayNum` (1–14) became the ten
   `holiday_is_*` fields; and — new in this version — `statusEnum` (0–3) is *also* now
   exposed as one-hot input fields (`freeflow_status`, `heavy_status`,
   `congested_status`, `imposible_status`), per `MachineLearningSpeedOfSectionView.java`
   and `REST-opperations.md`. Since every categorical the model needs is already
   one-hot on the wire, `model.py::TabularMLPRegressor` has **no embedding tables at
   all** — every earlier version of this model (`MK-1` through `MK-3`) needed at least
   one `nn.Embedding`.
2. **The task retargeted the model back to regressing `speed`** (as `MK-1`/`MK-2` did),
   this time with the status one-hot fields as *inputs* rather than the target. This is
   the reverse of `MK-3`'s design decision to exclude `statusEnum` for being
   near-circular with `speed` — see "Why this task is not really that hard" below for
   what that means for the achievable ceiling.

## Data contract

See `REST-opperations.md` in the repo root for the authoritative field-by-field
description of `/api/v1/sections/{id}/ml-speed-data`. Summary of what this model reads:

| Field(s) | Type | Role | Notes |
|---|---|---|---|
| `speed` | number (km/h) | **target** | |
| `dayNr` | int 0–6 | input | ISO weekday, Monday=0; cyclically encoded |
| `monthOfYear` | int 1–12 | input | cyclically encoded |
| `minutesSincDaybreak` | int 0–1439 | input | minutes since 06:00, wraps past midnight; cyclically encoded |
| `daysUntilHoliday` | int | input | standardized (zero mean, unit variance) |
| `holiday_type_regular_day`, `holiday_type_eve`, `holiday_type_holiday` | 0/1 | input | one-hot, exactly one is 1 |
| `freeflow_status`, `heavy_status`, `congested_status`, `imposible_status` | 0/1 | input | one-hot encoding of `statusEnum` — **derived from `speed`**, see below |
| `holiday_is_nyar` … `holiday_is_trettondag` (10 fields) | 0/1 | input | one-hot, exactly one is 1 |
| `sectionId`, `measureTime`, `status`, `statusEnum` | — | read, ignored | present in the export but not used as model inputs (`statusEnum`/`status` are redundant with the `*_status` one-hot fields already used) |

Rows with a null value in any required field are dropped up front
(`dataset.py::load_dataframe`). `dataset.py::REQUIRED_FIELDS` is exactly this table's
input + target columns, so pointing the pipeline at data still in the older
`holidayType`/`holidayNum`/no-one-hot-`statusEnum` shape fails fast with a clear
`ValueError` naming the missing fields, rather than silently training on the wrong
schema.

**Measured on the real export** (423,771 qualifying rows, collected
2026-09-19): `speed` ranges 4–80 km/h, mean 56.2, std 16.1. The `*_status` fields'
underlying class balance (freeflow 86.4%, heavy 10.7%, congested 2.7%, impossible
0.26%) is the same kind of skew `MK-3`'s Solution.md measured — it matters here too,
just differently (see next section).

## Why this task is not really that hard — and why that's by design, not a bug

The task explicitly lists `freeflow_status`/`heavy_status`/`congested_status`/
`imposible_status` among the model's *input* features. Those fields are a one-hot
encoding of `statusEnum`, which the upstream collector derives from `speed` itself via
fixed thresholds (see `SpeedOfSectionMLData.java`'s status bands — freeflow/heavy/
congested/impossible are literally non-overlapping speed ranges). Knowing which speed
band a reading falls into is therefore very strong information about the speed itself,
even before looking at any time-of-day/calendar covariate.

Concretely, on the real export, per-`statusEnum` speed statistics are:

| `statusEnum` | mean speed (km/h) | std (km/h) | rows |
|---|---|---|---|
| freeflow (0) | 60.8 | 11.4 | 366,207 |
| heavy (1) | 30.3 | 8.5 | 45,228 |
| congested (2) | 15.8 | 2.9 | 11,252 |
| impossible (3) | 8.5 | 1.4 | 1,084 |

A trivial model that ignores every other feature and just predicts the training-set
mean speed for the row's status (`E[speed | statusEnum]`) already achieves **MAE =
8.90 km/h, RMSE = 10.99 km/h, R² = 0.535** on the full dataset — computed directly from
the table above, no learning involved. This is the **ceiling** this model's non-status
features (`dayNr`, `monthOfYear`, `minutesSincDaybreak`, `daysUntilHoliday`,
`holiday_type_*`, `holiday_is_*`) would need to beat to be pulling their weight.

**Measured result: every configuration tried lands within noise of that ceiling, not
meaningfully above it.** See "Measured results" below — architecture (1–4 layers,
32–256 wide), activation, learning rate, and `--log-target` were all varied, and test
R² never rose above 0.532. The practical conclusion is that in the current data window,
time-of-day and calendar information carries almost no *additional* speed signal once
`statusEnum` is already known — the model has essentially learned "predict the
status-conditional mean," which is the best any model, however large, can do if that
really is where nearly all the signal lives. This is worth stating plainly because it's
easy to misread a good-looking R² here as "the MLP learned rich time/calendar
patterns" when it mostly reflects the status one-hot fields doing the work. See
"Suggested improvements" for how to test this more directly (e.g. an ablation with the
`*_status` fields removed) and why the task's inclusion of `*_status` as an input is
still reasonable despite this.

## Feature engineering and input normalization

`features.py` / `dataset.py`:

- **One-hot passthrough** (`features.py::one_hot_features`): the 17 upstream
  `holiday_type_*` / `*_status` / `holiday_is_*` fields are already 0/1, so they are
  concatenated into the feature vector unchanged — no embedding, no further scaling.
- **Cyclical features**: `dayNr`, `monthOfYear`, and `minutesSincDaybreak` are each
  encoded as `(sin, cos)` pairs, so e.g. minute 1439 and minute 0 are seen as adjacent
  rather than maximally distant. Carried over unchanged from `MK-2`/`MK-3`.
- **Scaled continuous feature**: `daysUntilHoliday` is standardized with
  `sklearn.preprocessing.StandardScaler`, **fit on the training split only**
  (`dataset.py::fit_scalers`) to avoid test-set leakage — the task's "normalize input
  data" requirement.
- **Target normalization**: `speed` is optionally `log1p`-transformed
  (`--log-target`, the task's "select if target variable should be transformed with
  log or not" switch) and then always standardized (`dataset.py::TargetTransform`),
  fit on the training split only. `evaluate.py` and `train.py`'s reported MAE/RMSE
  always invert this back to km/h, so `--log-target` changes what the *loss* optimizes,
  not the units results are reported in.

The full model input is a 24-wide vector: 17 one-hot columns + 7 cyclical/scaled
columns (`features.py::FEATURE_DIM`).

## Model (`model.py::TabularMLPRegressor`)

```
holiday_type_* (3), *_status (4), holiday_is_* (10)  ──────────────┐
                                                                      ├─ concat ──► (B, 24)
dayNr, monthOfYear, minutesSincDaybreak (sin/cos × 3)                │
daysUntilHoliday (scaled)                            ────────────────┘
                                                                      │
                                            [ Linear(→hidden_dim) → activation → Dropout ]
                                                      × --num-layers
                                                                      │
                                                                      ▼
                                                Linear(hidden_dim → 1)   (scaled speed)
                                                                      │
                                                                      ▼
                                          TargetTransform.inverse_transform → km/h
```

- `--num-layers` (default 2) sets the number of hidden layers; `0` collapses the model
  to plain linear regression on the 24-wide feature vector.
- `--hidden-dim` (default 64) sets the width of every hidden layer.
- `--activation` selects the nonlinearity used after every hidden `Linear`, per the
  task's required switch: `sigmoid`, `tanh` (hyperbolic tangent), `softsign`, `relu`,
  or `leaky_relu` (see `model.py::ACTIVATIONS`).
- `--dropout` (default 0.1) is applied after each hidden layer's activation.

### Weight initialization ("random initialization of weights")

`TabularMLPRegressor._init_weights` initializes every `nn.Linear` layer's weights from
a random distribution matched to the chosen activation — Kaiming-normal for
`relu`/`leaky_relu`, Xavier-normal for `sigmoid`/`tanh`/`softsign` — with biases at
zero, applied once at construction after `set_seed(args.seed)` has already seeded
`torch.manual_seed`. Reproducible per `--seed`, differs meaningfully between seeds.

## Training paradigms (`train.py`)

The task's "Features to implement" section names five ML paradigms; here is where each
lives in the code:

| Paradigm | Implementation |
|---|---|
| Gradient descent | `torch.optim.SGD` (not Adam — SGD is literally gradient descent, optionally with momentum) |
| Backpropagation | `loss.backward()` in `run_epoch`, followed by `nn.utils.clip_grad_norm_` and `optimizer.step()` |
| MiniBatch | `DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)` |
| Early stopping | `--early-stopping-patience` / `--early-stopping-min-delta`: training halts once validation MAE (km/h) hasn't *decreased* by at least `min_delta` for `patience` consecutive epochs (`patience=0` disables it) |
| Adjustable learning rate and momentum | `--lr`, `--momentum` (both passed straight to `SGD`); `--lr-decay` additionally applies a per-epoch multiplicative decay via `torch.optim.lr_scheduler.ExponentialLR` (`1.0` = constant LR) |
| Random initialization of weights | see above |

Loss is `nn.MSELoss` on the (optionally log1p'd, standardized) target. Gradient
clipping (`--grad-clip`, default 5.0) guards against occasional large gradients
destabilizing SGD, which — unlike Adam — has no built-in per-parameter adaptive scaling
to absorb them.

## Train/test/validation split

Per the task spec, unchanged from earlier versions:
`sklearn.model_selection.train_test_split(row_indices, test_size=0.1, random_state=4711,
shuffle=True)` (`dataset.py::split_train_test`), applied to plain row indices since
there is no windowing or per-segment grouping to respect. A further slice of the 90%
training partition (`--val-fraction`, default 0.1 of *that* partition, same seed) is
held out from gradient updates for early stopping / model selection only; scalers are
fit on the remaining train-only rows, and the mandated 10% test set is touched only by
the final `evaluate.py` run.

On the real export (423,771 qualifying rows): train=343,253, val=38,140, test=42,378.

## Measured results

All runs below use the real export (423,771 rows, collected 2026-09-19) on an M2
MacBook's Apple GPU (`--device mps`, auto-selected).

### Baseline (default architecture)

`--num-layers 2 --hidden-dim 64 --activation relu`, up to 30 epochs,
`--early-stopping-patience 8`: early-stopped at epoch 11, best val MAE 8.85 km/h.

```
Test set (n=42378): MAE=8.87 km/h  RMSE=11.01 km/h  MAPE=18.96%  R2=0.531
```

This is essentially identical to the status-conditional-mean ceiling computed above
(MAE 8.90, RMSE 10.99, R² 0.535) — see "Why this task is not really that hard."

### Architecture doesn't move the needle

`--num-layers 3 --hidden-dim 128 --activation leaky_relu --lr 3e-2
--early-stopping-patience 12`: early-stopped at epoch 23, best val MAE 8.82 km/h.

```
Test set (n=42378): MAE=8.84 km/h  RMSE=11.01 km/h  MAPE=18.97%  R2=0.532
```

Nearly 3x the parameters and 3x the learning rate produce a test MAE 0.03 km/h better
than the two-layer baseline — within run-to-run noise, not a real improvement.

### `--log-target` comparison

Same baseline architecture, up to 30 epochs, `--early-stopping-patience 8`:

| `--log-target` | Stopped at epoch | Best val MAE | Test MAE | Test RMSE | Test MAPE | Test R² |
|---|---|---|---|---|---|---|
| off (default) | 11 | 8.85 | **8.87** | **11.01** | 18.96% | **0.531** |
| on | 11 | 9.04 | 9.06 | 11.04 | **18.72%** | 0.529 |

`log1p` compresses large speed values more than small ones, so the model trained on
the log-transformed target is nudged toward relatively more accurate predictions at low
speeds — it wins on MAPE (a relative-error metric) but loses slightly on MAE/RMSE
(absolute-error metrics), the same kind of metric-choice trade-off `MK-2`'s Solution.md
observed. Neither setting is "wrong"; pick based on whether the downstream consumer
cares more about absolute km/h error or relative/percentage error. `--log-target` is
off by default because the task frames MAE-style absolute error as the more natural
metric for a speed prediction.

### Activation function comparison

`--num-layers 2 --hidden-dim 64`, up to 20 epochs, `--early-stopping-patience 6`:

| Activation | Stopped at epoch | Best val MAE |
|---|---|---|
| sigmoid | 13 | 8.90 |
| tanh | 15 | 8.83 |
| softsign | 15 | 8.81 |
| relu | 9 | 8.85 |
| leaky_relu | 9 | 8.85 |

Unlike `MK-3`'s classifier (where `sigmoid` was clearly worst and `relu`/`softsign`
led by a wide margin), **every activation lands within 0.1 km/h of every other one**
here. This is consistent with the "status one-hot dominates" finding above: when most
of the achievable fit is a near-linear function of the one-hot inputs, the choice of
hidden-layer nonlinearity has little room to matter. `--activation relu` remains the
default for consistency with the other `MK-*` versions and because it's the cheapest
to compute, not because it measurably wins here.

### Hyperparameter tuning (`tune.py`)

Random search over `--num-layers`, `--hidden-dim`, `--activation`, `--dropout`,
`--lr`, `--momentum`, `--lr-decay`, `--batch-size`, and `--log-target`. A short run (6
trials, up to 6 epochs each, `--early-stopping-patience 3`) on the real export:

| Trial | layers | hidden | activation | log_target | stopped@ | val MAE |
|---|---|---|---|---|---|---|
| 2 (winner) | 1 | 256 | sigmoid | True | 5 | **8.93** |
| 4 | 1 | 32 | leaky_relu | False | 6 | 8.94 |
| 1 | 2 | 256 | tanh | False | 4 | 8.96 |
| 3 | 3 | 64 | softsign | True | 6 | 9.03 |
| 6 | 2 | 128 | leaky_relu | True | 6 | 9.10 |
| 5 | 4 | 256 | tanh | True | 6 | 9.15 |

All six trials land within 0.22 km/h of each other, and none beats the baseline's 8.85
— reinforcing that, on this data, hyperparameter choice matters far less than it did for
`MK-3`'s imbalanced classification target. Run `tune.py` with a larger `--trials` /
`--trial-epochs` budget once more calendar diversity has accumulated in the export (see
"Suggested improvements"); a short search on a data window with little seasonal
variation mostly measures noise.

## Evaluation (`evaluate.py`)

Loads a checkpoint, rebuilds the *same* test split deterministically (same seed, read
back from the checkpoint's saved training args — the split itself isn't persisted), and
reports MAE, RMSE, MAPE, and R² (`checkpoints/eval/metrics.csv`), plus a predicted-vs-
true scatter plot and a residual histogram (`checkpoints/eval/prediction_plots.png`).

## Suggested improvements

**Modeling**
- **Ablate the `*_status` one-hot fields.** The "Why this task is not really that hard"
  section above is an indirect argument (comparing measured performance to a
  hand-computed ceiling); the direct test is to retrain with those four fields removed
  from `features.ONE_HOT_FIELDS`/`STATUS_FIELDS` and see how much test R² actually
  drops. If it drops close to zero, the model would be doing the genuinely harder
  "predict speed from time/calendar alone" task `MK-3`'s classifier variant already
  explored (there, from the other direction: predicting `statusEnum` from time/calendar
  alone). That may be a more useful model for a live congestion *predictor* than one
  that needs to already know the current status to predict the current speed.
- **Add `sectionId` as an embedding feature** (as `MK-1` does) — still likely the
  single highest-leverage change for the *within-status* residual signal this version
  leaves on the table. A motorway's freeflow speed and a suburban street's freeflow
  speed differ meaningfully; without segment identity the model can only learn one
  global status-conditional mean averaged across all sections.
- Now that the model's ceiling is well understood, a natural next experiment is
  training two model variants side by side — with and without `*_status` — and
  comparing not just their headline metrics but their residual distributions, to see
  whether the "with-status" model is doing anything beyond conditional-mean
  prediction at all.

**Optimization**
- `SGD` was used as specified by the task's "gradient descent" paradigm requirement.
  Given how close every tried configuration already sits to the status-conditional-mean
  ceiling, further optimizer tuning (Adam, a learning-rate schedule beyond
  `--lr-decay`'s fixed exponential decay) is unlikely to move test metrics much until
  the ablation above changes what the model is actually trying to learn.

**Data / evaluation caveats**
- **The current `raw-trafic-data.json` export is not yet calendar-diverse.** It spans a
  short recent window, so `monthOfYear` and the holiday fields carry little variation
  — the model has almost no opportunity to learn seasonal effects yet, which likely
  understates how much the non-status features could contribute once more calendar
  range has accumulated. Re-run the comparisons above periodically as the export grows.
- A block-in-time split (e.g. most recent N days held out) would better measure genuine
  forecasting skill once enough calendar range exists; the random 90/10 split (as
  specified in the task) is fine for now since rows are i.i.d. with no leakage risk, but
  it never tests generalization to an unseen time period the way a temporal holdout
  would.
- The `impossible` status band has only 1,084 rows (≈0.26%) — its contribution to any
  aggregate metric (MAE, R²) is small simply by row count, independent of how well the
  model handles it specifically.

**Engineering**
- No learning-rate scheduler more sophisticated than a fixed exponential decay
  (`--lr-decay`) is implemented; `ReduceLROnPlateau` (decay only when validation MAE
  stalls) is a natural next step if a future ablation gives SGD a harder optimization
  landscape to work with than the current status-dominated one.
