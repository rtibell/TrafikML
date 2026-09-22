# Solution design — MachineLearning_MK-5

A PyTorch multilayer perceptron (MLP) that **classifies** the traffic status of one
road-segment measurement into one of four classes — `freeflow_status`, `heavy_status`,
`congested_status`, `imposible_status` — from the non-circular `SegmentSpeed`
covariates: `dayNr`, `daysUntilHoliday`, `minutesSincDaybreak`, `monthOfYear`, and 13
upstream one-hot fields (`holiday_type_*`, `holiday_is_*`).

This is the fifth iteration of the modelling approach in this repository:

| Version | Task | Architecture | Categorical inputs |
|---|---|---|---|
| `MK-1` | regress `speed` | LSTM/GRU sequence model, `sectionId` embedded | per-segment minute-by-minute time series |
| `MK-2` | regress `speed` | 1D-CNN over a concatenated feature vector | `statusEnum`/`holidayType`/`holidayNum` embedded |
| `MK-3` | classify `statusEnum` | MLP, softmax output | `holidayType`/`holidayNum` embedded |
| `MK-4` | regress `speed` | MLP | none — upstream one-hot fields fed directly (incl. `*_status` as an *input*) |
| **`MK-5` (this version)** | **classify the `*_status` one-hot group** | **MLP, softmax output** | **none — upstream `holiday_type_*`/`holiday_is_*` one-hot fields fed directly** |

## What changed from `MK-4`, and why

The task retargeted the model from `MK-4`'s regression of `speed` back to
classification — but of the **status one-hot fields** (`freeflow_status`,
`heavy_status`, `congested_status`, `imposible_status`) specifically, rather than the
integer `statusEnum` column `MK-3` used. Two consequences follow:

- **The `*_status` fields move from inputs (`MK-4`) to the target (this version).**
  Since they can only take the value 0 or 1 and exactly one of the four is 1 per row,
  they encode a single 4-way categorical variable, not four independent binary labels
  — see "One softmax head, not four independent sigmoids" below for why that matters
  for the model's output layer and loss function.
- **`speed` is not added as an input.** It would be the obvious substitute input, but
  the status fields are *derived from* `speed` by the upstream collector (see
  `SpeedOfSectionMLData.java`'s status thresholds — freeflow/heavy/congested/impossible
  are literally speed bands). Using `speed` to predict status would make the task close
  to circular (a simple threshold classifier could do it, no learning required) — the
  same reasoning `MK-3`'s Solution.md gave for excluding `speed` there. The model's
  input feature set is therefore `dayNr`, `daysUntilHoliday`, `minutesSincDaybreak`,
  `monthOfYear`, and the 13 `holiday_type_*`/`holiday_is_*` one-hot fields — everything
  in the revised `SegmentSpeed` structure except `speed`, `sectionId`, `measureTime`,
  `status`, `statusEnum`, and the four `*_status` target fields themselves.

This is effectively `MK-3`'s classification problem again (predict congestion status
from time/calendar covariates alone, with no direct observation of current traffic
conditions), now expressed against the revised one-hot schema and consequently with **no
embedding tables at all** — `holiday_type_*` and `holiday_is_*` arrive pre-one-hot on
the wire, so unlike `MK-3` (which embedded `holidayType`/`holidayNum` from small
integers) there is nothing left to embed.

## One softmax head, not four independent sigmoids

The four `*_status` fields are mutually exclusive (`statusEnum` in the same payload
confirms this: it's a single 0–3 integer, and exactly one `*_status` field is 1 to
match it). That makes this a **single categorical variable with 4 possible values**,
one-hot encoded on the wire — not four independent binary attributes a row could have
any combination of.

Two architectures were available:

1. **Four independent sigmoid outputs + binary cross-entropy per field** — the natural
   choice if the fields really were independent binary labels (multi-label
   classification).
2. **One softmax output over 4 classes + categorical cross-entropy** — the natural
   choice for a single mutually-exclusive categorical variable (multi-class
   classification), which is what this actually is.

This version uses **option 2**, unchanged from `MK-3`'s approach and consistent with
the task's original "use a softmax to determine the status value" framing. Concretely:
`dataset.py::derive_status_class` collapses the four one-hot columns back into one
integer class index (`argmax`, in `features.STATUS_VALUES` order: freeflow=0, heavy=1,
congested=2, impossible=3) once at data-loading time, and everything downstream —
`model.py`, `train.py`'s `nn.CrossEntropyLoss`, `evaluate.py`'s confusion matrix — is
exactly `MK-3`'s single-label 4-class classifier. Four independent sigmoids would
*work* in the sense of producing four probabilities, but would (a) let the model
predict combinations that can't occur in the data (e.g. both `freeflow_status=1` and
`heavy_status=1`), wasting model capacity and training signal enforcing a constraint
softmax gives for free, and (b) need an extra un-normalized "which one wins" tie-break
step to recover a single predicted status, duplicating what `argmax(softmax(logits))`
already does in one line. Softmax is the more honest match to what the target actually
is.

## Data contract

See `REST-opperations.md` in the repo root for the authoritative field-by-field
description of `/api/v1/sections/{id}/ml-speed-data`. Summary of what this model reads:

| Field(s) | Type | Role | Notes |
|---|---|---|---|
| `freeflow_status`, `heavy_status`, `congested_status`, `imposible_status` | 0/1 | **target** | one-hot, exactly one is 1; collapsed to a single class index 0–3 (`dataset.py::derive_status_class`) |
| `dayNr` | int 0–6 | input | ISO weekday, Monday=0; cyclically encoded |
| `monthOfYear` | int 1–12 | input | cyclically encoded |
| `minutesSincDaybreak` | int 0–1439 | input | minutes since 06:00, wraps past midnight; cyclically encoded |
| `daysUntilHoliday` | int | input | standardized (zero mean, unit variance) |
| `holiday_type_regular_day`, `holiday_type_eve`, `holiday_type_holiday` | 0/1 | input | one-hot, exactly one is 1, fed through unchanged |
| `holiday_is_nyar` … `holiday_is_trettondag` (10 fields) | 0/1 | input | one-hot, exactly one is 1, fed through unchanged |
| `sectionId`, `measureTime`, `status`, `statusEnum`, `speed` | — | read, ignored | present in the export but not used — `statusEnum`/`status` are redundant with the target fields already used, and `speed` is excluded for the circularity reason above |

Rows with a null value in any required field are dropped up front
(`dataset.py::load_dataframe`). `dataset.py::REQUIRED_FIELDS` is exactly this table's
input + target columns, so pointing the pipeline at data missing the one-hot fields
(e.g. an export predating the schema change, or `MK-3`'s old integer
`holidayType`/`holidayNum`/`statusEnum`-only shape) fails fast with a clear
`ValueError` naming the missing fields.

### Class balance — the central challenge of this task

**Measured on the real export** (438,847 rows, collected 2026-09-20):

| Class | Count | Share |
|---|---|---|
| freeflow (0) | 381,237 | 86.9% |
| heavy (1) | 45,274 | 10.3% |
| congested (2) | 11,252 | 2.6% |
| impossible (3) | 1,084 | 0.25% |

A model that always predicts `freeflow` scores **~86.9% raw accuracy** on the held-out
test set while never once identifying a congested or impossible road — see "Why
balanced accuracy, not raw accuracy" below. Everything about loss weighting, the
model-selection metric, and evaluation reporting in this version is shaped by this
imbalance, exactly as it was in `MK-3`.

## Feature engineering and input normalization

`features.py` / `dataset.py`:

- **One-hot passthrough** (`features.py::one_hot_features`): the 13 upstream
  `holiday_type_*`/`holiday_is_*` fields are already 0/1, so they are concatenated
  into the feature vector unchanged — no embedding, no further scaling.
- **Cyclical features**: `dayNr`, `monthOfYear`, and `minutesSincDaybreak` are each
  encoded as `(sin, cos)` pairs, so e.g. minute 1439 and minute 0 are seen as adjacent
  rather than maximally distant. Carried over unchanged from earlier versions.
- **Scaled continuous feature**: `daysUntilHoliday` is standardized with
  `sklearn.preprocessing.StandardScaler`, **fit on the training split only**
  (`dataset.py::fit_scalers`) to avoid test-set leakage — the task's "normalize input
  data" requirement.

The full model input is a 20-wide vector: 13 one-hot columns + 7 cyclical/scaled
columns (`features.py::FEATURE_DIM`).

## Model (`model.py::TabularMLPClassifier`)

```
holiday_type_* (3), holiday_is_* (10)                ──────────────┐
                                                                      ├─ concat ──► (B, 20)
dayNr, monthOfYear, minutesSincDaybreak (sin/cos × 3)                │
daysUntilHoliday (scaled)                            ────────────────┘
                                                                      │
                                            [ Linear(→hidden_dim) → activation → Dropout ]
                                                      × --num-layers
                                                                      │
                                                                      ▼
                                                Linear(hidden_dim → 4)   (raw logits)
                                                                      │
                                                                      ▼
                                              softmax → argmax → predicted status class
```

- `--num-layers` (default 2) sets the number of hidden layers; `0` collapses the model
  to plain (linear + softmax) multinomial logistic regression on the 20-wide feature
  vector.
- `--hidden-dim` (default 64) sets the width of every hidden layer.
- `--activation` selects the nonlinearity used after every hidden `Linear`: `sigmoid`,
  `tanh` (hyperbolic tangent), `softsign`, `relu`, or `leaky_relu` (see
  `model.py::ACTIVATIONS`).
- `--dropout` (default 0.1) is applied after each hidden layer's activation.

### Softmax: where it lives and why

`forward()` returns **raw logits** (one per class), not softmax-normalized
probabilities. Training minimizes `nn.CrossEntropyLoss`, which internally combines
`log_softmax` and negative-log-likelihood in one numerically stable operation —
applying a separate `nn.Softmax` before it would double-normalize and needlessly lose
precision (a well-known PyTorch pitfall). The softmax is applied explicitly at
**inference/prediction time** instead, in `model.py::predict_status`:

```python
def predict_status(logits):
    probs = torch.softmax(logits, dim=-1)   # <- the softmax
    return probs.argmax(dim=-1), probs      # <- argmax picks the predicted status class
```

Every accuracy/metric computation in `train.py::run_epoch` and `evaluate.py` calls
this function rather than re-deriving the logic, so there is exactly one place softmax
happens.

### Weight initialization ("random initialization of weights")

`TabularMLPClassifier._init_weights` initializes every `nn.Linear` layer's weights from
a random distribution matched to the chosen activation — Kaiming-normal for
`relu`/`leaky_relu`, Xavier-normal for `sigmoid`/`tanh`/`softsign` — with biases at
zero, applied once at construction after `set_seed(args.seed)` has already seeded
`torch.manual_seed`. Reproducible per `--seed`, differs meaningfully between seeds.

## Why balanced accuracy, not raw accuracy, drives training and evaluation

This is the most consequential design decision in this version, unchanged from `MK-3`:

With **unweighted** `CrossEntropyLoss` and **raw accuracy** as the early-stopping/
model-selection metric, the model converges to always predicting `freeflow`. Two
changes fix this:

1. **Class-weighted loss** (`--class-weights balanced`, the default):
   `dataset.py::compute_class_weights` computes `n_samples / (n_classes * count[c])`
   per class from the training split, and passes it to `nn.CrossEntropyLoss(weight=...)`
   so a mistake on a rare `impossible` row costs the loss much more than a mistake on a
   common `freeflow` row.
2. **Balanced accuracy** (macro-average recall across the 4 classes, computed in
   `train.py::run_epoch`) replaces raw accuracy as the early-stopping and
   best-checkpoint criterion. A model that always predicts one class scores exactly
   `1/4 = 0.25` on this metric regardless of that class's frequency.

**Measured on the real export**, default architecture, up to 60 epochs with
`--early-stopping-patience 8`:

| `--class-weights` | Stopped at epoch | Best val balanced acc. |
|---|---|---|
| `none` (unweighted) | 9 | 0.2500 |
| `balanced` (default) | 14 | **0.4794** |

The unweighted model's balanced accuracy of exactly 0.2500 confirms it degenerated to
predicting a single class every time (train/val raw accuracy ≈ 86.8%, matching the
freeflow share exactly). `--class-weights balanced` is the default for this reason.

## Training paradigms (`train.py`)

| Paradigm | Implementation |
|---|---|
| Gradient descent | `torch.optim.SGD` (not Adam — SGD is literally gradient descent, optionally with momentum) |
| Backpropagation | `loss.backward()` in `run_epoch`, followed by `nn.utils.clip_grad_norm_` and `optimizer.step()` |
| MiniBatch | `DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)` |
| Early stopping | `--early-stopping-patience` / `--early-stopping-min-delta`: training halts once validation *balanced accuracy* hasn't improved by at least `min_delta` for `patience` consecutive epochs (`patience=0` disables it) |
| Adjustable learning rate and momentum | `--lr`, `--momentum` (both passed straight to `SGD`); `--lr-decay` additionally applies a per-epoch multiplicative decay via `torch.optim.lr_scheduler.ExponentialLR` (`1.0` = constant LR) |
| Random initialization of weights | see above |

Gradient clipping (`--grad-clip`, default 5.0) guards against occasional large
gradients destabilizing SGD, which — unlike Adam — has no built-in per-parameter
adaptive scaling to absorb them; this matters more here than in a regression version
since class-weighted cross-entropy on a rare class can produce a much larger
per-sample gradient than an unweighted loss would.

## Train/test/validation split

Per the task spec, unchanged from earlier versions:
`sklearn.model_selection.train_test_split(row_indices, test_size=0.1, random_state=4711,
shuffle=True)` (`dataset.py::split_train_test`), applied to plain row indices since
there is no windowing or per-segment grouping to respect. A further slice of the 90%
training partition (`--val-fraction`, default 0.1 of *that* partition, same seed) is
held out from gradient updates for early stopping / model selection only; the feature
scaler is fit on the remaining train-only rows, and the mandated 10% test set is
touched only by the final `evaluate.py` run.

On the real export (438,847 qualifying rows): train=355,465, val=39,497, test=43,885.

## Measured results

All runs below use the real export (438,847 rows, collected 2026-09-20) on an M2
MacBook's Apple GPU (`--device mps`, auto-selected).

### Baseline (default architecture, `--class-weights balanced`)

`--num-layers 2 --hidden-dim 64 --activation relu`, up to 60 epochs,
`--early-stopping-patience 8`: early-stopped at epoch 14, best val balanced accuracy
0.4794.

```
Test set (n=43885): accuracy=0.5440  balanced_accuracy=0.4763  macro-F1=0.2553
class        precision    recall        f1   support
freeflow         0.941     0.586     0.722     38215
heavy            0.139     0.202     0.165      4456
congested        0.058     0.447     0.103      1093
impossible       0.015     0.669     0.030       121
```

Reading this: the model correctly flags roughly 45% of `congested` rows and 67% of
`impossible` rows using *only* time-of-day and calendar information — no current speed
observation at all — at the cost of many false positives on those classes (their
precision is low: most "congested"/"impossible" *predictions* are actually a milder
status). Whether that trade-off is useful depends on the downstream consumer: a system
that treats "predicted congested/impossible" as "worth a closer look" tolerates false
positives far better than one that acts on the prediction directly.

### Activation function comparison

`--num-layers 2 --hidden-dim 64 --class-weights balanced`, up to 20 epochs,
`--early-stopping-patience 6`:

| Activation | Stopped at epoch | Best val balanced accuracy |
|---|---|---|
| sigmoid | 13 | 0.3410 |
| tanh | 20 (budget, still improving) | **0.4927** |
| softsign | 20 (budget, still improving) | 0.4785 |
| relu | 12 | 0.4794 |
| leaky_relu | 12 | 0.4826 |

As in `MK-3`, `sigmoid` is clearly worst; the other four cluster within ~0.02 of each
other, with `tanh` and `leaky_relu` leading (and `tanh`/`softsign` not yet converged
within the 20-epoch budget — a longer run would likely move them further). Sigmoid's
tendency to saturate (squashing large-gradient signals toward its flat regions) appears
to hurt more under a class-weighted loss's larger, more variable per-sample gradients
for rare classes. `--activation relu` remains the default for consistency with the
other `MK-*` versions; `tanh`/`leaky_relu` are worth trying if squeezing out the last
few points of balanced accuracy matters more than matching the family default.

### Hyperparameter tuning (`tune-rnd.py` / `tune-ga.py`)

Two interchangeable tuners search the same 9-hyperparameter space (`--num-layers`,
`--hidden-dim`, `--activation`, `--dropout`, `--lr`, `--momentum`, `--lr-decay`,
`--batch-size`, `--class-weights`) and reuse `train.py::run_training` for every
candidate, exactly as the original single-script `tune.py` did before this version
split it in two:

- **`tune-rnd.py`** — uniform random search, the original approach. Kept unmodified as
  a comparison baseline.
- **`tune-ga.py`** — a genetic algorithm (GA), using
  [`PyGAD`](https://github.com/ahmedfgad/GeneticAlgorithmPython) (open-source,
  BSD-3-Clause). A population of candidate configurations evolves over several
  generations via fitness-proportionate selection, crossover, and mutation, instead of
  every trial being sampled independently of every other trial's result. Chosen over
  the other common free/open-source option, DEAP, because PyGAD's `gene_space`
  parameter maps directly onto this search: each hyperparameter's discrete choice list
  becomes one gene's allowed values, so PyGAD's built-in operators already respect them
  with no hand-written chromosome encoding or custom mutation logic needed.

Both rank candidates by the same metric `train.py` itself optimizes: validation
*balanced* accuracy. `tune-ga.py` additionally logs macro precision/F1/F2 per candidate
(computed via a small `train.py::run_epoch` extension that now also tracks per-class
predicted counts, so these come "for free" from the same epoch loop already running for
early stopping — no extra forward pass). See `Tuning.md` for the full design
walkthrough (chromosome encoding, GA operator choices and why, fitness caching) and
tuning tips.

**Measured comparison** (20,000-row synthetic dataset — not yet the real export, see
`Tuning.md`'s caveat — 20 random-search trials vs. an 8-individual/3-generation GA run,
identical training budget):

| | Random search (`tune-rnd.py`) | GA (`tune-ga.py`) |
|---|---|---|
| Best val balanced accuracy | **0.8781** (trial 10/20) | 0.8612 (generation 2) |
| Test balanced accuracy (winner retrained) | **0.6345** | 0.6039 |
| Test macro-F1 | **0.5575** | 0.5368 |
| Degenerate (~0.33, single-class-collapse) trials | 9 of 20 (45%) | 0 of 8 by generation 2 |
| Population/trial mean fitness trend | flat, no learning across trials | **rising every generation: 0.41 → 0.53 → 0.58 → 0.62** |

Random search's single best trial narrowly won this particular small-budget comparison
— reported here plainly rather than as a GA win, since it wasn't one. What the GA
demonstrably delivered instead: its population's *mean* fitness climbed every
generation (direct evidence of directed search, not blind sampling) and it never
produced a degenerate individual past generation 1, while nearly half of random
search's independent trials collapsed to the useless ~0.33 floor by chance. A GA's
advantage compounds with more generations; `Tuning.md` recommends a larger
population/generation budget (and averaging multiple `--search-seed` runs) before
drawing a final conclusion on a production-scale dataset — this run used a small
budget specifically to keep the comparison's wall-clock short. Full findings, per-
generation tables, and the confusion-matrix-level test comparison are in `Tuning.md`.

## Evaluation (`evaluate.py`)

Loads a checkpoint, rebuilds the *same* test split deterministically (same seed, read
back from the checkpoint's saved training args — the split itself isn't persisted),
and reports accuracy, balanced accuracy, macro-F1, a full per-class precision/recall/F1
table, and a confusion matrix (as text, CSV, and a row-normalized heatmap PNG).

## Suggested improvements

**Modeling**
- **Add `sectionId` as an embedding feature** (as `MK-1` does) — still the single
  highest-leverage change available. Congestion timing and severity differ enormously
  by segment (a motorway on-ramp vs. a quiet suburban street), and without segment
  identity the model can only learn one global time-of-day/calendar congestion curve
  averaged across all sections.
- This remains a genuinely non-circular prediction task (time/calendar covariates
  only, no current speed observation), which is good, but it also means the ceiling on
  achievable accuracy is inherently lower than a model that got to see `speed` or the
  status fields as inputs (compare `MK-4`, which — by using the status fields as
  *inputs* to regress `speed` — converges near a much higher, easier-to-reach ceiling;
  see its Solution.md). Read the ~48% balanced accuracy measured here against that
  ceiling, not against the ~87% raw accuracy of a model that cheats by ignoring rare
  classes.
- Try focal loss (down-weights easy, well-classified examples and up-weights hard
  ones) as an alternative or complement to class-weighted cross-entropy.

**Optimization**
- `SGD` was used as specified by the task's "gradient descent" paradigm requirement.
  Class-weighted cross-entropy produces larger, more variable per-sample gradients for
  rare classes than an unweighted loss, which shows up as noisy epoch-to-epoch
  validation balanced accuracy in the training logs above — a lower `--lr`, a smaller
  `--grad-clip` value, or averaging the last few epochs' checkpoints instead of taking
  a single best-epoch snapshot would all likely reduce this noise.
- `--lr-decay` is exposed but not exercised by the measured comparisons above (all used
  the default `1.0`). Both `tune-rnd.py`'s and `tune-ga.py`'s searches do sample it.
- **Re-run `tune-ga.py` with a larger population/generation budget against the real
  export.** The measured comparison above used a small budget (8×3) specifically to
  keep it fast; see `Tuning.md`'s "Tips for tuning the tuner" for concrete next steps
  (population/generation sizing, mutation rate, multi-seed averaging).

**Data / evaluation caveats**
- **The current `raw-trafic-data.json` export is not yet calendar-diverse.** It spans a
  short recent window, so `monthOfYear` and the holiday fields carry little variation
  — today's reported metrics are a pipeline-correctness check, not a
  production-readiness measure. Re-run tuning and evaluation once more calendar
  diversity has accumulated.
- A block-in-time split (e.g. most recent N days held out) would better measure genuine
  forecasting skill once enough calendar range exists; the random 90/10 split (as
  specified in the task) is fine for now since rows are i.i.d. with no leakage risk, but
  it never tests generalization to an unseen time period the way a temporal holdout
  would.
- The `impossible` class has only 1,084 examples total (≈108 expected in the 10% test
  split) — any single evaluation's `impossible` precision/recall is a small-sample
  estimate with wide variance; don't over-read small differences in that row between
  runs.

**Engineering**
- No learning-rate scheduler more sophisticated than a fixed exponential decay
  (`--lr-decay`) is implemented; `ReduceLROnPlateau` (decay only when validation
  balanced accuracy stalls) would likely reduce the epoch-to-epoch noise mentioned
  above and is a natural next step.
- `evaluate.py`'s per-class precision/recall/F1 uses the sklearn `zero_division=0`
  convention (a class the model never predicts scores 0, not undefined) specifically so
  a degenerate always-predict-majority model can't look good by having its other
  classes' undefined scores silently excluded from a `nanmean`-style average.
