# Operations — running MachineLearning_MK-3

How to set up, get data, train, tune, and evaluate the MLP `statusEnum` classifier. See
`Solution.md` for design rationale and measured results.

## Setup

```bash
cd MachineLearning_MK-3
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Device is picked automatically, in this priority order (see `train.py::select_device`):
1. **CUDA** (e.g. a GTX 1080), if `torch.cuda.is_available()`.
2. **Apple GPU (Metal/MPS)**, e.g. an M2 MacBook, if `torch.backends.mps.is_available()`.
3. **CPU**, otherwise.

Override with `--device cpu`, `--device cuda:0`, or `--device mps` if you need to force
a specific one. `train.py`/`tune.py`/`evaluate.py` all print which device they picked.
This model is a small MLP over a 17-wide input; on an M2 MacBook it trains real data
(≈157k training rows) in under 2 seconds per epoch on the Apple GPU, and is fast enough
on plain CPU too — GPU mainly helps by letting `--batch-size`, `--hidden-dim`, and
`tune.py`'s trial count scale up cheaply.

## Getting data

The pipeline expects `raw-trafic-data.json` in this directory: a JSON array of
`SegmentSpeed` records (see `Solution.md`'s Data contract section for the exact
fields). Only `statusEnum` (the target) and the six non-`speed` covariates are used;
`speed` itself, if present, is read but ignored — see Solution.md for why.

### Option A — export from a running backend (recommended)

```bash
# with the Spring Boot backend running (e.g. ./gradlew bootRun --args='--spring.profiles.active=dev')
python export_raw_trafic_data.py --base-url http://localhost:8080 --out raw-trafic-data.json
```

Stdlib-only (no venv needed to run it); walks every section from
`GET /api/v1/sections`, pulls each one's `/ml-speed-data`, and writes a single valid
JSON array. Identical to `MachineLearning_MK-1`/`MK-2`'s script of the same name.

> **Do not** use the `curl ... | jq | sed 's/]/,/' >> raw-trafic-data.json` pattern
> from `gen_raw-trafic-data.sh` at the repo root — it only fetches one section per
> invocation and the `sed` trick does not produce valid, closed JSON.

### Option B — synthetic data (smoke test, no backend needed)

```bash
python generate_synthetic_data.py --out synthetic-trafic-data.json --rows 20000
```

Fabricates independent rows with a rough rush-hour speed/status profile (no
per-segment continuity is needed here — see Solution.md on why this model treats rows
as i.i.d.). Useful to confirm the pipeline runs (dataset → model → train → tune →
evaluate) before pointing it at real data. Writes to `synthetic-trafic-data.json` by
default — **never** point it at `raw-trafic-data.json`. Its holiday/calendar fields are
simplified stand-ins, not the real Swedish holiday calendar `SwedishHolidays.java`
computes, and its class balance is not as skewed as the real export's.

## Training

```bash
python train.py --data raw-trafic-data.json --epochs 60 --num-layers 2 --activation relu
```

Key flags (all optional; `python train.py --help` for the full list):

| Flag | Default | Meaning |
|---|---|---|
| `--val-fraction` | 0.1 | held out from the 90% train split, for model selection only |
| `--batch-size` | 256 | minibatch size (the task's "MiniBatch" paradigm) |
| `--epochs` | 100 | **maximum** epochs — early stopping may end training sooner |
| `--num-layers` | 2 | number of hidden layers in the MLP; `0` = linear + softmax (multinomial logistic regression) |
| `--hidden-dim` | 64 | width of each hidden layer |
| `--activation` | `relu` | `sigmoid` \| `tanh` \| `softsign` \| `relu` \| `leaky_relu` |
| `--dropout` | 0.1 | applied after every hidden layer's activation |
| `--holiday-type-embed-dim` / `--holiday-num-embed-dim` | 2 / 4 | categorical embedding sizes |
| `--class-weights` | `balanced` | `balanced` weights cross-entropy inversely to training-set class frequency; `none` disables this — see Solution.md for why `balanced` is the default on this heavily-imbalanced target |
| `--lr` | 1e-2 | initial SGD learning rate |
| `--momentum` | 0.9 | SGD momentum |
| `--lr-decay` | 1.0 | multiplicative per-epoch LR decay (`ExponentialLR`); `1.0` = constant LR |
| `--grad-clip` | 5.0 | max gradient norm |
| `--early-stopping-patience` | 10 | stop after this many epochs with no *validation balanced accuracy* improvement of at least `--early-stopping-min-delta`; `0` disables early stopping |
| `--early-stopping-min-delta` | 1e-4 | minimum val balanced-accuracy improvement to reset the patience counter |
| `--seed` | 4711 | seeds Python/NumPy/PyTorch (including weight init) and the train/test/val split |
| `--device` | auto | override auto-detected `cuda`/`mps`/`cpu` |
| `--checkpoint-dir` | `checkpoints` | where checkpoints/history are written |

`train.py` prints the selected device and the train/val/test row counts before
training starts, then one line per epoch with the current LR, and train/val loss,
raw accuracy, and **balanced accuracy** (macro-average recall — the metric that
actually drives early stopping/checkpointing; see Solution.md for why raw accuracy
alone is misleading on this imbalanced target). Expect balanced accuracy to be noisier
epoch-to-epoch than the loss, especially with `--class-weights balanced` — see
Solution.md's "Optimization" suggestions if this is a problem.

**Outputs** (in `--checkpoint-dir`):
- `best.pt` — checkpoint with the highest validation *balanced* accuracy seen so far (use this for evaluation)
- `last.pt` — checkpoint from the final epoch (equal to `best.pt` only if the last epoch was also the best one)
- `history.json` — per-epoch train/val loss, accuracy, balanced accuracy, and per-class recall, for plotting learning curves

Each checkpoint bundles the model weights, the training args (needed to reconstruct
the model architecture and re-derive the same data split), and the fitted
`daysUntilHoliday` scaler — `evaluate.py` needs nothing else.

### Trying each activation function

```bash
for act in sigmoid tanh softsign relu leaky_relu; do
    python train.py --data raw-trafic-data.json --activation "$act" \
        --checkpoint-dir "checkpoints-$act"
done
```

See Solution.md's "Activation function comparison" for measured results on the real
export — `relu` wins clearly here, unlike the earlier regression version of this model.

### Comparing class-weighted vs. unweighted loss directly

```bash
python train.py --data raw-trafic-data.json --checkpoint-dir /tmp/balanced
python train.py --data raw-trafic-data.json --class-weights none --checkpoint-dir /tmp/unweighted
python evaluate.py --checkpoint /tmp/balanced/best.pt   --data raw-trafic-data.json --output-dir /tmp/balanced/eval
python evaluate.py --checkpoint /tmp/unweighted/best.pt --data raw-trafic-data.json --output-dir /tmp/unweighted/eval
```

Compare the two runs' printed accuracy/balanced-accuracy/macro-F1 lines and confusion
matrices — the unweighted run should collapse to predicting `freeflow` for every row
(balanced accuracy exactly 0.25, three all-zero confusion-matrix rows).

## Hyperparameter tuning

```bash
python tune.py --data raw-trafic-data.json --trials 20 --trial-epochs 20 --final-epochs 100
```

Random search over `--num-layers`, `--hidden-dim`, `--activation`, `--dropout`, `--lr`,
`--momentum`, `--lr-decay`, `--batch-size`, the two embedding dims, and
`--class-weights`, seeded by `--search-seed` (default 4711 — see `Solution.md`). Every
trial reuses `train.py`'s early-stopping-aware training loop; the winner (by best
validation **balanced** accuracy) is retrained for up to `--final-epochs` and
checkpointed to `<output-dir>/best_checkpoint/`.

**Outputs** (in `--output-dir`, default `tuning/`):
- `tuning_results.csv` — every trial's config, validation balanced accuracy, and the epoch it stopped at
- `best_config.json` — the winning trial's config
- `best_checkpoint/best.pt` — the retrained winning model, usable directly with `evaluate.py`

Increase `--trials` for a more thorough search once you have GPU time to spend; the
per-trial cost is one early-stopping-bounded `run_training` call, so search cost scales
with `--trials × (average epochs per trial)`, not `--trials × --trial-epochs` directly
(most trials stop well before the cap).

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best.pt --data raw-trafic-data.json
```

Prints accuracy, balanced accuracy, macro-F1, and a full per-class precision/recall/F1
table on the held-out 10% test split, and writes:
- `checkpoints/eval/metrics.csv` — the same numbers, one row per class plus summary rows
- `checkpoints/eval/confusion_matrix.csv` — the raw (true × predicted) counts
- `checkpoints/eval/confusion_matrix.png` — a row-normalized confusion-matrix heatmap

**Important**: the test split is *recomputed*, not stored — `evaluate.py` rebuilds it
from `--data` (or the path stored in the checkpoint's training args, if `--data` is
omitted) using the same seed. This only reproduces the original held-out rows if
`raw-trafic-data.json` is unchanged since training. If you've re-exported or appended
to `raw-trafic-data.json` since training a checkpoint, evaluate against a copy of the
file as it was at training time, or retrain.

## End-to-end smoke test

A quick way to confirm everything is wired correctly before spending real GPU time:

```bash
python generate_synthetic_data.py --out synthetic-trafic-data.json --rows 10000
python train.py --data synthetic-trafic-data.json --epochs 10 --batch-size 128 \
    --num-layers 2 --hidden-dim 32 --checkpoint-dir /tmp/mk3-smoke
python evaluate.py --checkpoint /tmp/mk3-smoke/best.pt --data synthetic-trafic-data.json \
    --output-dir /tmp/mk3-smoke/eval
```

This was run as part of building this version (8,000 synthetic rows) on CPU, MPS
(`--device mps`), and with auto-detection, and every `--activation` choice trained
successfully. It was also run against the real `raw-trafic-data.json` export (194,401
rows, on an M2 MacBook's Apple GPU): training converged (with early stopping) in
9–22 epochs at ~1.8–2s/epoch depending on configuration, and `evaluate.py` produced
sensible, if modest, classification metrics — see Solution.md's "Measured results" for
the actual numbers. Delete `/tmp/mk3-smoke` and `synthetic-trafic-data.json`
afterwards; they're scratch output.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `ValueError: raw-trafic-data.json is missing expected fields` | File doesn't match the `SegmentSpeed` schema — check field names, especially `holidayType` vs. the API's `holidayNr` |
| `json.decoder.JSONDecodeError` on load | File is truncated/invalid JSON — likely built with the old `curl \| jq \| sed` approach; re-export with `export_raw_trafic_data.py` |
| Raw accuracy looks great (>85%) but balanced accuracy is ~0.25 | The model has collapsed to always predicting `freeflow` — check `--class-weights` is `balanced` (the default) and that early stopping is watching balanced accuracy, not raw accuracy (it does by default; don't average/override `run_training`'s selection logic) |
| Val balanced accuracy swings wildly between epochs | Expected to some degree with class-weighted loss on rare classes (see Solution.md) — try a lower `--lr`, a lower `--grad-clip`, or a longer `--early-stopping-patience` so a temporary dip doesn't stop training early |
| Training stops after only a few epochs | Early stopping fired — check `checkpoints/history.json`; if it stopped too early, raise `--early-stopping-patience` or lower `--early-stopping-min-delta` |
| `impossible`/`congested` precision or recall is 0 or looks unstable across runs | These classes have very little support (526 and 4,735 rows total respectively, out of 194,401) — see Solution.md's caveat on small-sample metrics for the rarest classes |
| `CUDA out of memory` / Metal-related OOM on MPS | Lower `--batch-size` (unlikely to matter for this model's size, but possible with a very large `--hidden-dim`) |
| `evaluate.py` metrics look implausible after re-exporting data | `raw-trafic-data.json` changed since training (different sections/date range, different class balance) — retrain, or point `--data` at the original export |
