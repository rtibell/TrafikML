# Solution design — MachineLearning_MK-1

Traffic speed/status forecaster for the Stockholm-region road segments collected by
the TrafficML backend. Given `t_in` minutes of a segment's recent history, it predicts
speed (km/h) and congestion status for the next `t_out` minutes.

## Requested stack vs. what's implemented

The task asked for PyTorch Geometric or DGL "on top of PyTorch." Both are **graph**
neural network libraries — their value is spatial message-passing between segments
that share an edge (e.g. a T-GCN / DCRNN / Graph WaveNet architecture: a graph
convolution over the road network's adjacency, feeding a GRU/LSTM per node). That
requires a segment *adjacency graph*, which the `SegmentSpeed` records don't carry —
there's no geometry/topology field, only `sectionId`. `DefinitionOfSection` in the
Java backend does have GeoJSON geometry that could be used to derive adjacency
(segments whose endpoints touch or lie within some distance), which would be the
natural next step (see "Future work" below) — but building and validating that graph
was out of scope without first seeing a plain sequence model work.

So this version implements the *other* architecture the task named — "Recurrent
sequence models per segment (LSTM/GRU or seq2seq) ... one model with the segment ID as
an embedding feature" — in plain PyTorch, no PyG/DGL. It's written so a graph layer can
be dropped in later: `model.py`'s `CovariateEmbeddings.segment` is the seam where a
GNN-produced per-segment vector would replace the plain lookup embedding.

## Why one shared model, not one model per segment

The task named both as options. With thousands of segments (see README.md §1 of the
main project), training and serving a separate model per segment doesn't scale, and
segments with sparse or noisy history get no benefit from what similar segments
learned. A single model with `sectionId` folded in as an embedding (`nn.Embedding`)
shares the recurrent weights across every segment while still letting the model
specialize per segment through the embedding vector — the standard way to add an
entity id to a sequence model.

## Why seq2seq subsumes "LSTM/GRU"

The task listed "LSTM/GRU" and "seq2seq" as alternatives. Setting `--t-out 1` collapses
the implemented encoder-decoder to a single decoder step — i.e. exactly a plain
one-step LSTM/GRU forecaster. Rather than maintain two model implementations, one
covers both: `t_out=1` for a simple forecaster, `t_out>1` for genuine multi-step
forecasting.

## Data contract

Input is a JSON array of `SegmentSpeed` records, one per (`sectionId`, minute).
Field names follow the task's literal schema (`holidayType` is the same field the
`/api/v1/sections/{id}/ml-speed-data` REST endpoint calls `holidayNr` — see
`REST-opperations.md` in the repo root for full field semantics):

| Field | Type | Notes |
|---|---|---|
| `sectionId` | int | Foreign key into `DefinitionOfSection`; mapped to a dense embedding index (`SegmentIndex`), not used as a raw number |
| `measureTime` | ISO datetime | Must be exactly 1-minute cadence within a contiguous run |
| `status` / `statusEnum` | string / int 0–3 | `freeflow=0, heavy=1, congested=2, impossible=3` |
| `speed` | float, nullable | km/h; null rows are dropped before windowing |
| `dayNr` | int 0–6 | ISO weekday, Monday=0; cyclically encoded |
| `monthOfYear` | int 1–12 | cyclically encoded |
| `minutesSincDaybreak` | int 0–1439 | minutes since 06:00, wraps past midnight; cyclically encoded |
| `daysUntilHoliday` | int | scaled (standardized) |
| `holidayType` | int 0–2 | regular/eve/holiday; embedded |
| `holidayNum` | int 0–14 | which named holiday is next; embedded (0 unused/reserved) |

Records with a null `speed` or `statusEnum` are dropped up front. See
`Operations.md` for how the raw file is produced.

## Windowing

Each training example is one `(t_in history minutes -> t_out forecast minutes)` slice
for a single segment (`dataset.py::build_windows`). A window is only kept if its full
`t_in + t_out` span is *strictly consecutive* one-minute measurements — computed by
diffing timestamps within each segment's sorted rows and cutting at any gap ("run"
detection in `_windows_for_group`). A segment with a data gap simply contributes fewer
windows; no imputation is attempted. `--stride` controls how far the window start
advances between consecutive samples from the same segment (default 5 minutes, i.e.
windows overlap heavily by design — see the train/test split caveat below).

## Feature engineering

- **Cyclical features** (`features.py::cyclical_encode`): `dayNr`, `monthOfYear`, and
  `minutesSincDaybreak` are each encoded as `(sin, cos)` pairs so the model sees, e.g.,
  minute 1439 and minute 0 as adjacent rather than maximally distant — a plain integer
  or one-hot would tell the model midnight and 23:59 are unrelated.
- **Scaled continuous features**: `speed` and `daysUntilHoliday` are standardized
  (zero mean, unit variance) with `sklearn.preprocessing.StandardScaler`, **fit on the
  training split only** (`fit_scalers`, called after the train/test split) to avoid
  test-set leakage into the normalization statistics.
- **Categorical embeddings**: `statusEnum` (4), `holidayType` (3), `holidayNum` (15,
  0 reserved) each get a small learned embedding table, shared between encoder and
  decoder (`model.py::CovariateEmbeddings`) so "freeflow" or "holidayType=1" means the
  same vector wherever it appears.
- **Known-future vs. unknown-future covariates**: calendar/holiday fields are
  deterministic functions of the timestamp, so they're available for the *decoder's*
  future timesteps too (`calendar_features` is called once over the full
  `t_in + t_out` span and sliced). Only `speed` and `statusEnum` are actually unknown
  for future minutes — those are the two targets, fed to the decoder autoregressively.

## Model (`model.py`)

```
segment_idx ──► segment embedding ──────────────────────────┐
                                                              │ (broadcast every step)
encoder inputs (t_in steps):                                 │
  speed_scaled, status_emb, holiday_type_emb,                ▼
  holiday_num_emb, calendar(sin/cos×3 + days_scaled)  ──►  GRU/LSTM encoder ──► final hidden state
                                                                                       │
decoder inputs (t_out steps, one at a time):                                          ▼
  prev_speed_scaled, prev_status_emb (teacher-forced                         GRU/LSTM decoder
  or model's own last prediction), future calendar/                                  │
  holiday covariates, segment embedding            ◄──────────────────────────────────┘
                                                              │
                                                    ┌─────────┴─────────┐
                                                    ▼                   ▼
                                            speed_head (Linear)   status_head (Linear)
                                            → speed (km/h)        → 4-way logits
```

- `make_rnn` selects `nn.GRU` or `nn.LSTM` via `--cell`; both encoder and decoder use
  the same cell type.
- The decoder runs one step at a time in a Python loop (`Seq2SeqTrafficForecaster.forward`)
  because each step's input depends on the previous step's output (autoregressive).
- **Teacher forcing**: during training, at each decoder step there's a
  `teacher_forcing_ratio` chance of feeding the *true* previous speed/status instead of
  the model's own prediction — stabilizes early training when predictions are still
  poor, at the cost of a train/inference mismatch if never decayed (see Future work).
  At inference (`model.eval()`) or when no targets are passed, it's always fully
  autoregressive.
- **Loss**: `MSE(speed_scaled) + status_loss_weight * CrossEntropy(status_logits)`,
  summed over all `t_out` steps and averaged over the batch — a single joint
  optimization for both heads, sharing the encoder/decoder weights.

## Train/test split

Per the task spec: all windows across all segments are pooled into one list, then
split via `sklearn.model_selection.train_test_split(windows, test_size=0.1,
random_state=4711, shuffle=True)` (`dataset.py::split_train_test`). A further slice of
the 90% training partition (`--val-fraction`, default 0.1 of *that* partition, same
seed) is held out from gradient updates for early-stopping/model-selection only — the
mandated 10% test set is never used for anything except the final `evaluate.py` run.

**Caveat worth knowing**: because windows overlap (stride 5 on a 60+15-minute window
means neighboring windows share ~93% of their timesteps), a random shuffle can and
does place near-duplicate windows on both sides of the train/test boundary. This
inflates reported test performance relative to genuine skill at forecasting *unseen
time periods* — a block-in-time split (e.g., most recent N days per segment held out)
would be the honest measure for a real deployment decision. Implemented as specified
in the task (seed 4711, shuffled, 90/10); see Future work.

## Evaluation (`evaluate.py`)

Loads a checkpoint, reconstructs the *same* windows/split deterministically (same
seed + `t_in`/`t_out`/`stride`, read back from the checkpoint's saved training args —
nothing about the split itself is persisted), runs the model fully autoregressively
(no teacher forcing, matching real deployment use), and reports per-horizon-minute
metrics: speed MAE/RMSE/MAPE (km/h) and status accuracy/macro-F1, plus overall
pooled numbers, a CSV, and a MAE/accuracy-vs-horizon plot.

## Future work

**Architecture / modeling**
- Build the segment adjacency graph from `DefinitionOfSection` geometry and add a
  GCN/GAT layer (PyTorch Geometric or DGL, as originally requested) before the
  encoder — e.g. a DCRNN or Graph WaveNet style model. Highest-value change: road
  segments are spatially correlated (a jam upstream predicts a jam downstream a few
  minutes later), and nothing in the current model can see that — it only knows
  "which segment" via an independent embedding, not "which segments are near it and
  what are *they* doing right now."
- Add attention over the encoder's hidden states (Bahdanau/Luong-style) instead of
  handing the decoder only the encoder's final hidden state — helps `t_out` beyond
  ~15–30 minutes, where a single fixed-size hidden vector becomes an information
  bottleneck.
- Curriculum-decay `--teacher-forcing-ratio` to 0 over training (`--tf-decay-to-zero`
  is implemented but off by default) — closes the train/inference distribution gap
  where the decoder never sees its own errors during training otherwise.
- Add exogenous features not in the current payload but plausibly available
  server-side: weather (precipitation/temperature), scheduled roadworks, and
  event/venue calendars — traffic status is driven by more than time-of-day and
  holidays.

**Data / training procedure**
- Switch to a time-based (block-in-time) train/test split for an honest measure of
  real-world forecasting skill — see the caveat above.
- Increase `--stride` or de-duplicate near-identical windows to reduce training set
  redundancy (training set size grows ~linearly as stride shrinks, with limited new
  information per additional window).
- Handle time-series gaps with interpolation instead of discarding spanning windows,
  once gap frequency in real data is known.
- Per-segment (or per-road-class) speed normalization instead of one global scaler —
  a residential street's "free flow" and a motorway's "free flow" are different
  scales; the segment embedding can partially compensate, but explicit normalization
  usually converges faster and generalizes better to segments with few samples.
- Class-weight or focal loss for the status head — `congested`/`impossible` are
  almost certainly rare relative to `freeflow`; watch macro-F1 (not just accuracy) on
  real data.

**Hyperparameters to sweep once real data is available**
- `--hidden-size` (64/128/256) and `--num-layers` (1–3) — current defaults (128, 2)
  are a reasonable starting point, not tuned.
- `--t-in` / `--t-out` — 60→15 minutes is a guess; tune against how far ahead the
  downstream congestion-prediction consumer actually needs.
- `--status-loss-weight` — currently 0.3; raise it if status accuracy matters more
  than speed MAE for the downstream use case.
- Learning-rate scheduling (`ReduceLROnPlateau` on val loss) and early stopping — not
  implemented; currently trains for a fixed `--epochs` and keeps the best-val
  checkpoint, but never stops early or decays the LR.

**Engineering**
- `TrafficWindowDataset.__getitem__` recomputes cyclical/holiday features per sample
  on the CPU at data-loading time; precomputing them once as vectorized DataFrame
  columns before windowing would cut per-item overhead, which matters more once
  `--num-workers > 0` is bottlenecked by Python-level per-item work.
- `evaluate.py`'s reliance on recomputing an identical split is fragile if
  `raw-trafic-data.json` changes between train and evaluate runs (new data appended
  shifts window indices for every downstream segment). Storing the actual test window
  keys (`sectionId`, window start timestamp) in the checkpoint would make evaluation
  robust to the source file growing over time.
