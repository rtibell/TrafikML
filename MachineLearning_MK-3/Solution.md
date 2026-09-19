# Solution design — MachineLearning_MK-3

A PyTorch multilayer perceptron (MLP) that **classifies** `statusEnum` — one of four
traffic-status classes (`freeflow=0, heavy=1, congested=2, impossible=3`) — for one
road-segment measurement, from the non-circular `SegmentSpeed` covariates: `dayNr`,
`daysUntilHoliday`, `holidayType`, `minutesSincDaybreak`, `monthOfYear`, `holidayNum`.

This is the third iteration of the modelling approach in this repository, and the
second revision of this directory specifically:

| Version | Task | Architecture | Input framing |
|---|---|---|---|
| `MK-1` | regress `speed` | LSTM/GRU sequence model, `sectionId` embedded | per-segment minute-by-minute time series |
| `MK-2` | regress `speed` | 1D-CNN over a concatenated feature vector | independent rows, no sequence |
| `MK-3` (v1) | regress `speed` | MLP over the same concatenated feature vector | independent rows, no sequence |
| **`MK-3` (this version)** | **classify `statusEnum`** | **MLP, softmax output** | independent rows, no sequence |

## What changed from `MK-3`'s first version, and why

The task was retargeted from regression (`speed`, km/h) to 4-class classification
(`statusEnum`). Two consequences follow directly from that:

- **`statusEnum` is removed from the inputs.** It was previously one of three embedded
  categorical inputs (alongside `holidayType`, `holidayNum`); now that it *is* the
  target, including it as an input would be predicting a value from itself.
- **`speed` is not added as an input.** It would be the obvious substitute input, but
  `statusEnum` is *derived from* `speed` by the upstream collector (see
  `SpeedOfSectionMLData.java`'s status thresholds — freeflow/heavy/congested/impossible
  are literally speed bands). Using `speed` to predict `statusEnum` would make the task
  close to circular (a simple threshold classifier could do it, no learning required),
  and defeats the point flagged by this project's own earlier Solution.md revisions:
  the genuinely useful "predict congestion status from time/calendar covariates alone"
  problem. The model's input feature set is therefore exactly the earlier feature list
  minus `statusEnum` and `speed`: `dayNr`, `daysUntilHoliday`, `holidayType`,
  `minutesSincDaybreak`, `monthOfYear`, `holidayNum`.

This is a harder, more honest problem than the regression version: predicting
congestion status from *only* time-of-day/calendar information, with no direct
observation of current traffic conditions at all.

## Data contract

Same `SegmentSpeed` JSON array as `MK-1`/`MK-2` (see `REST-opperations.md` in the repo
root for full field semantics; `holidayType` is the task's name for the API's
`holidayNr`). `speed` and `sectionId`/`measureTime`, if present in the export, are read
but ignored — see above for why `speed` specifically is excluded.

| Field | Type | Role | Notes |
|---|---|---|---|
| `statusEnum` | int 0–3 | **target** | `freeflow=0, heavy=1, congested=2, impossible=3` |
| `dayNr` | int 0–6 | input | ISO weekday, Monday=0; cyclically encoded |
| `monthOfYear` | int 1–12 | input | cyclically encoded |
| `minutesSincDaybreak` | int 0–1439 | input | minutes since 06:00, wraps past midnight; cyclically encoded |
| `daysUntilHoliday` | int | input | standardized (zero mean, unit variance) |
| `holidayType` | int 0–2 | input | regular/eve/holiday; embedded |
| `holidayNum` | int 0–14 | input | which named holiday is next; embedded (0 unused/reserved) |

Rows with a null value in any required field are dropped up front
(`dataset.py::load_dataframe`).

### Class balance — the central challenge of this task

**Measured on the real export** (194,401 rows, 100 sections):

| Class | Count | Share |
|---|---|---|
| freeflow (0) | 170,513 | 87.7% |
| heavy (1) | 18,627 | 9.6% |
| congested (2) | 4,735 | 2.4% |
| impossible (3) | 526 | 0.3% |

A model that always predicts `freeflow` scores **87.9% raw accuracy** on the held-out
test set while never once identifying a congested or impossible road — see "Why
balanced accuracy, not raw accuracy" below. Everything about loss weighting, the
model-selection metric, and evaluation reporting in this version is shaped by this
imbalance.

## Feature engineering and input normalization

Same pipeline as before (`features.py`, `dataset.py`), minus the `statusEnum`
embedding:

- **Cyclical features**: `dayNr`, `monthOfYear`, and `minutesSincDaybreak` are each
  encoded as `(sin, cos)` pairs, so e.g. minute 1439 and minute 0 are seen as adjacent
  rather than maximally distant.
- **Scaled continuous feature**: `daysUntilHoliday` is standardized with
  `sklearn.preprocessing.StandardScaler`, **fit on the training split only**
  (`dataset.py::fit_scalers`) to avoid test-set leakage — the task's "normalize input
  data" requirement.
- **Categorical embeddings**: `holidayType` (3), `holidayNum` (15, 0 reserved) each get
  a small learned `nn.Embedding` table, sized by `--holiday-type-embed-dim`,
  `--holiday-num-embed-dim`.

## Model (`model.py::TabularMLPClassifier`)

```
holidayType ──► embedding (dim=holiday_type_embed_dim) ┐
holidayNum ──► embedding (dim=holiday_num_embed_dim)    ├─ concat ──┐
                                                                      │
dayNr, monthOfYear, minutesSincDaybreak (sin/cos ×3)     concat ──► (B, input_dim)
daysUntilHoliday (scaled)                ──────────────┘             │
                                                                      ▼
                                            [ Linear(→hidden_dim) → activation → Dropout ]
                                                      × --num-layers
                                                                      │
                                                                      ▼
                                                Linear(hidden_dim → 4)   (raw logits)
                                                                      │
                                                                      ▼
                                              softmax → argmax → predicted statusEnum
```

- `--num-layers` (default 2) sets the number of hidden layers; `0` collapses the model
  to plain (linear + softmax) multinomial logistic regression on the engineered
  feature vector.
- `--hidden-dim` (default 64) sets the width of every hidden layer.
- `--activation` selects the nonlinearity used after every hidden `Linear`, per the
  task's required switch: `sigmoid`, `tanh` (hyperbolic tangent), `softsign`, `relu`,
  or `leaky_relu` (see `model.py::ACTIVATIONS`).
- `--dropout` (default 0.1) is applied after each hidden layer's activation.

### Softmax: where it lives and why

The task requires "use a softmax to determine statusEnum value." `forward()` returns
**raw logits** (one per class), not softmax-normalized probabilities. Training
minimizes `nn.CrossEntropyLoss`, which internally combines `log_softmax` and
negative-log-likelihood in one numerically stable operation — applying a separate
`nn.Softmax` before it would double-normalize and needlessly lose precision (a
well-known PyTorch pitfall). The softmax the task asks for is applied explicitly at
**inference/prediction time** instead, in `model.py::predict_status`:

```python
def predict_status(logits):
    probs = torch.softmax(logits, dim=-1)   # <- the softmax
    return probs.argmax(dim=-1), probs      # <- argmax picks the predicted statusEnum
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

This is the most consequential design decision in this version, discovered empirically
while building it (see "Measured results" below for the numbers that motivated it):

With **unweighted** `CrossEntropyLoss` and **raw accuracy** as the early-stopping/
model-selection metric, the model converges to always predicting `freeflow` — 87.9%
accuracy, but 0% recall on all three other classes. Raw accuracy *rewards* this,
because it can't tell "genuinely skillful" apart from "correctly guessed the majority
class every time." Two changes fix this:

1. **Class-weighted loss** (`--class-weights balanced`, the default):
   `dataset.py::compute_class_weights` computes `n_samples / (n_classes * count[c])`
   per class from the training split (the standard "balanced" weighting), and passes
   it to `nn.CrossEntropyLoss(weight=...)` so a mistake on a rare `impossible` row
   costs the loss much more than a mistake on a common `freeflow` row.
2. **Balanced accuracy** (macro-average recall across the 4 classes, computed in
   `train.py::run_epoch`) replaces raw accuracy as the early-stopping and
   best-checkpoint criterion. A model that always predicts one class scores exactly
   `1/4 = 0.25` on this metric regardless of that class's frequency, so it can no
   longer look good by ignoring the rare classes. `evaluate.py` reports both metrics
   (plus macro-F1 and a full per-class precision/recall/F1 table and confusion matrix)
   so the trade-off is always visible, not hidden behind one headline number.

**Measured on the real export**, default architecture, up to 60 epochs with
`--early-stopping-patience 8`:

| `--class-weights` | Stopped at epoch | Best val balanced acc. | Test accuracy | Test balanced accuracy | Test macro-F1 |
|---|---|---|---|---|---|
| `none` (unweighted) | 9 | 0.2500 | **0.8785** | **0.2500** | 0.2338 |
| `balanced` (default) | 15 | 0.4521 | 0.6201 | **0.4419** | **0.2857** |

The unweighted model's balanced accuracy of exactly 0.2500 confirms it degenerated to
predicting a single class every time (its confusion matrix has three all-zero rows).
The balanced-weighted model trades a much lower raw accuracy for meaningfully better
performance across all four classes — see the confusion matrix in "Measured results."
`--class-weights balanced` is the default for this reason, even though it looks
"worse" by the naive accuracy number alone.

## Training paradigms (`train.py`)

The task's "Features to implement" section names five ML paradigms; here is where each
lives in the code:

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
adaptive scaling to absorb them; this matters more here than in the earlier regression
version, since class-weighted cross-entropy on a rare class can produce a much larger
per-sample gradient than an unweighted regression loss ever did.

## Train/test/validation split

Per the task spec, unchanged from earlier versions:
`sklearn.model_selection.train_test_split(row_indices, test_size=0.1, random_state=4711,
shuffle=True)` (`dataset.py::split_train_test`), applied to plain row indices since
there is no windowing or per-segment grouping to respect. A further slice of the 90%
training partition (`--val-fraction`, default 0.1 of *that* partition, same seed) is
held out from gradient updates for early stopping / model selection only; the feature
scaler is fit on the remaining train-only rows, and the mandated 10% test set is
touched only by the final `evaluate.py` run.

On the real export (194,401 qualifying rows, 100 sections): train=157,464,
val=17,496, test=19,441.

## Activation function comparison

**Measured on the real export**, all other hyperparameters held at the default
architecture (`--num-layers 2 --hidden-dim 64 --lr 1e-2 --momentum 0.9 --batch-size
256 --class-weights balanced`), up to 30 epochs with `--early-stopping-patience 6`:

| Activation | Stopped at epoch | Best val balanced accuracy |
|---|---|---|
| sigmoid | 19 | 0.3569 |
| tanh | 19 | 0.4358 |
| softsign | 22 | 0.4710 |
| relu | 22 | **0.4757** |
| leaky_relu | 13 | 0.4482 |

Unlike the earlier regression version of this model (where saturating activations
matched or slightly beat ReLU at this shallow depth), `sigmoid` is clearly worst here
and `relu`/`softsign` lead. Classification with a class-weighted loss produces some
much larger per-sample gradients (for the rare classes) than plain MSE regression did;
`sigmoid`'s tendency to saturate (squashing large-gradient signals toward its flat
regions) appears to hurt more under that gradient profile. `--activation relu` is the
default for this reason.

## Hyperparameter tuning (`tune.py`)

Random search (not grid) over `--num-layers`, `--hidden-dim`, `--activation`,
`--dropout`, `--lr`, `--momentum`, `--lr-decay`, `--batch-size`, the two embedding
dims, and `--class-weights`, seeded by `--search-seed` (default 4711, same constant as
the data split, but a logically separate seed). Every trial reuses
`train.py::run_training` directly (including early stopping and balanced-accuracy
model selection), so tuning and standalone training always run identical code. Each
trial trains for up to `--trial-epochs`; the winning configuration (by best validation
*balanced* accuracy — never raw accuracy, for the reason given above) is retrained for
up to `--final-epochs` and checkpointed.

**Search results on the real export** (12 trials, up to 20 epochs each with
`--early-stopping-patience 5`, final retrain up to 60 epochs; full trial table in
`tuning/tuning_results.csv` after running `tune.py`):

| Trial | layers | hidden | activation | dropout | lr | momentum | batch | class_weights | stopped@ | val balanced acc. |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 (winner) | 4 | 256 | relu | 0.0 | 3e-2 | 0.0 | 1024 | balanced | 20 (still improving) | **0.4774** |
| 4 | 2 | 256 | leaky_relu | 0.2 | 3e-2 | 0.95 | 256 | balanced | 8 | 0.4233 |
| 3 | 2 | 128 | leaky_relu | 0.1 | 3e-3 | 0.0 | 128 | balanced | 20 | 0.4268 |
| 9 | 3 | 128 | softsign | 0.3 | 3e-3 | 0.8 | 128 | balanced | 20 | 0.4221 |
| 1, 5, 6, 11 (worst, tied) | — | — | — | — | — | — | — | **none** | 6 | 0.2500 |

The four `class_weights=none` trials all scored *exactly* 0.2500 — every one of them
degenerated to predicting a single class, confirming the "Why balanced accuracy, not
raw accuracy" analysis above holds across architectures, not just the one baseline
config. Every `class_weights=balanced` trial scored well above that floor. The winning
trial (10) hit its 20-epoch budget without early stopping firing — it may not have
fully converged; a larger `--trial-epochs` would let deeper/wider configs like this one
(4 layers, 256-wide) show their true ceiling instead of being cut off.

Retraining trial 10's config for up to 60 epochs (it ran 18 before early stopping) and
evaluating on the held-out test set:

```
Test set (n=19441): accuracy=0.5592  balanced_accuracy=0.4639  macro-F1=0.2612
class        precision    recall        f1   support
freeflow         0.948     0.602     0.736     17079
heavy            0.157     0.213     0.181      1842
congested        0.062     0.333     0.105       462
impossible       0.011     0.707     0.022        58
```

Compared to the untuned baseline (accuracy 0.6201, balanced accuracy 0.4419, macro-F1
0.2857), the tuned model trades a lower raw accuracy and macro-F1 for a higher balanced
accuracy — driven mostly by a much higher `impossible` recall (0.707 vs. 0.345), at the
cost of that class's already-poor precision falling further (0.011: for every correct
`impossible` prediction, roughly 90 are false alarms). This is the same kind of
metric-choice trade-off `MK-2`'s tuning run showed for MAE-vs-RMSE: optimizing
`tune.py`'s ranking criterion (balanced accuracy) doesn't automatically optimize every
other reasonable metric (macro-F1, raw accuracy) at the same time. Whether the tuned or
untuned model is "better" depends on whether the downstream use case values catching
rare severe-congestion events (favor the tuned model) or minimizing false alarms
(favor the baseline, or rank `tune.py` by macro-F1 instead).

## Evaluation (`evaluate.py`)

Loads a checkpoint, rebuilds the *same* test split deterministically (same seed, read
back from the checkpoint's saved training args — the split itself isn't persisted),
and reports accuracy, balanced accuracy, macro-F1, a full per-class precision/recall/F1
table, and a confusion matrix (as text, CSV, and a row-normalized heatmap PNG).

**Baseline result** (default architecture, `--class-weights balanced`,
early-stopped at epoch 15):

```
Test set (n=19441): accuracy=0.6201  balanced_accuracy=0.4419  macro-F1=0.2857
class        precision    recall        f1   support
freeflow         0.940     0.661     0.776     17079
heavy            0.137     0.286     0.185      1842
congested        0.083     0.476     0.141       462
impossible       0.022     0.345     0.041        58
```

Reading this: the model correctly flags roughly half of `congested` rows (0.476
recall) and a third of `impossible` rows (0.345 recall) using *only* time-of-day and
calendar information — no current speed observation at all — at the cost of many false
positives on those classes (their precision is low: most "congested" *predictions* are
actually freeflow). Whether that trade-off is useful depends on the downstream
consumer: a system that treats "predicted congested/impossible" as "worth a closer
look" tolerates false positives far better than one that acts on the prediction
directly.

## Suggested improvements

**Modeling**
- **Add `sectionId` as an embedding feature** (as `MK-1` does) — still the single
  highest-leverage change available. Congestion timing and severity differ enormously
  by segment (a motorway on-ramp vs. a quiet suburban street), and without segment
  identity the model can only learn one global time-of-day/calendar congestion curve
  averaged across all 100 sections.
- **This is now a genuinely non-circular prediction task** (unlike the earlier
  regression version's near-circular `statusEnum`-from-`speed` concern), which is good,
  but it also means the ceiling on achievable accuracy is inherently lower: time and
  calendar alone can capture *typical* congestion patterns (rush hour, weekday vs.
  weekend) but not *incidental* congestion (an accident, an event, weather) that the
  collector's `speed` measurement would show immediately. The measured ~44% balanced
  accuracy should be read against that ceiling, not against the ~88% raw accuracy of a
  model that cheats by ignoring rare classes.
- **Try focal loss** (`down-weights easy, well-classified examples and up-weights hard
  ones) as an alternative or complement to class-weighted cross-entropy — it's designed
  for exactly this kind of severe imbalance and sometimes outperforms simple inverse-
  frequency weighting.
- The activation comparison above used a fixed, shallow (`num_layers=2`) architecture;
  `tune.py`'s search varies both together and should be trusted over the isolated
  comparison once a full search has been run with more trials.

**Optimization**
- `SGD` was used as specified by the task's "gradient descent" paradigm requirement.
  Class-weighted cross-entropy produces larger, more variable per-sample gradients for
  rare classes than the earlier regression loss did, which shows up as noisier
  epoch-to-epoch validation balanced accuracy in the training logs above (jumping
  ±0.05–0.10 between consecutive epochs even near convergence) — a lower `--lr`, a
  smaller `--grad-clip` value, or averaging the last few epochs' checkpoints instead of
  taking a single best-epoch snapshot would all likely reduce this noise.
- `--lr-decay` is exposed but not exercised by the measured comparisons above (all used
  the default `1.0`). `tune.py`'s search does sample it.

**Data / evaluation caveats** (mostly unchanged from earlier versions, still apply)
- **The current `raw-trafic-data.json` export is not yet representative.**
  `holidayType`/`holidayNum` carry little variation and `monthOfYear` is nearly
  constant in the current export window, so the model can't yet learn holiday/seasonal
  effects — today's reported metrics are a pipeline-correctness check, not a
  production-readiness measure. Re-run tuning and evaluation once more calendar
  diversity has accumulated.
- A block-in-time split (e.g. most recent N days held out) would better measure genuine
  forecasting skill once enough calendar range exists; the random 90/10 split (as
  specified in the task) is fine for now since rows are i.i.d. with no leakage risk, but
  it never tests generalization to an unseen time period the way a temporal holdout
  would.
- The `impossible` class has only 526 examples total (≈53 expected in the 10% test
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
  classes' undefined scores silently excluded from a `nanmean`-style average — worth
  being aware of if extending `evaluate.py` further, since the more "obvious" `nan`-
  skipping approach is exactly what produces the misleadingly high macro-F1 this
  convention avoids.
