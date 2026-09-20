# Operations — running MachineLearning_MK-4

How to set up, get data, train, tune, and evaluate the MLP `speed` regressor. See
`Solution.md` for design rationale and measured results.

## Setup

```bash
cd MachineLearning_MK-4
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Device is picked automatically, in this priority order (see `train.py::select_device`):
1. **CUDA** (e.g. a GTX 1080), if `torch.cuda.is_available()`.
2. **Apple GPU (Metal/MPS)**, e.g. an M2 MacBook, if `torch.backends.mps.is_available()`.
3. **CPU**, otherwise.

Override with `--device cpu`, `--device cuda:0`, or `--device mps` if you need to force
a specific one. `train.py`/`tune.py`/`evaluate.py` all print which device they picked.
This model is a small MLP over a 24-wide input; on an M2 MacBook it trains real data
(≈343k training rows) in ~3s per epoch on the Apple GPU, and is fast enough on plain
CPU too — GPU mainly helps by letting `--batch-size`, `--hidden-dim`, and `tune.py`'s
trial count scale up cheaply.

## Getting data

The pipeline expects `raw-trafic-data.json` in this directory: a JSON array of
`SegmentSpeed` records matching the **revised** schema — see `Solution.md`'s Data
contract section, or `REST-opperations.md` in the repo root, for the exact fields.
This version's schema replaced the old small-integer `holidayType`/`holidayNum`
categoricals with one-hot fields, and additionally exposes `statusEnum` as one-hot
`*_status` fields.

> **If `raw-trafic-data.json` predates this schema change** (no `holiday_type_*` /
> `*_status` / `holiday_is_*` fields, or the old integer `holidayType`/`holidayNum`
> instead), `dataset.py::load_dataframe` raises `ValueError: ... is missing expected
> fields: [...]` immediately rather than silently training on the wrong columns.
> Re-export it (Option A below) once the backend's `/ml-speed-data` endpoint is
> serving the new shape.

### Option A — export from a running backend (recommended)

```bash
# with the Spring Boot backend running (e.g. ./gradlew bootRun --args='--spring.profiles.active=dev')
python export_raw_trafic_data.py --base-url http://localhost:8080 --out raw-trafic-data.json
```

Stdlib-only (no venv needed to run it); walks every section from
`GET /api/v1/sections`, pulls each one's `/ml-speed-data`, and writes a single valid
JSON array. Identical to `MachineLearning_MK-1`/`MK-2`/`MK-3`'s script of the same
name — it doesn't need to know or care about the field-schema change, since it just
forwards whatever JSON the backend returns.

> **Do not** use the `curl ... | jq | sed 's/]/,/' >> raw-trafic-data.json` pattern
> from `gen_raw-trafic-data.sh` at the repo root — it only fetches one section per
> invocation and the `sed` trick does not produce valid, closed JSON.

### Option B — synthetic data (smoke test, no backend needed)

```bash
python generate_synthetic_data.py --out synthetic-trafic-data.json --rows 20000
```

Fabricates independent rows with a rough rush-hour speed/status profile (no
per-segment continuity is needed here — see Solution.md on why this model treats rows
as i.i.d.), already in the revised one-hot schema. Useful to confirm the pipeline runs
(dataset → model → train → tune → evaluate) before pointing it at real data. Writes to
`synthetic-trafic-data.json` by default — **never** point it at `raw-trafic-data.json`.
Its holiday/calendar fields are simplified stand-ins, not the real Swedish holiday
calendar `SwedishHolidays.java` computes, and its class balance is not as skewed as the
real export's.

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
| `--num-layers` | 2 | number of hidden layers in the MLP; `0` = plain linear regression |
| `--hidden-dim` | 64 | width of each hidden layer |
| `--activation` | `relu` | `sigmoid` \| `tanh` \| `softsign` \| `relu` \| `leaky_relu` |
| `--dropout` | 0.1 | applied after every hidden layer's activation |
| `--log-target` | off | train on `log1p(speed)` instead of raw `speed` before standardizing — see Solution.md for the measured MAE-vs-MAPE trade-off |
| `--lr` | 1e-2 | initial SGD learning rate |
| `--momentum` | 0.9 | SGD momentum |
| `--lr-decay` | 1.0 | multiplicative per-epoch LR decay (`ExponentialLR`); `1.0` = constant LR |
| `--grad-clip` | 5.0 | max gradient norm |
| `--early-stopping-patience` | 10 | stop after this many epochs with no *validation MAE* improvement of at least `--early-stopping-min-delta`; `0` disables early stopping |
| `--early-stopping-min-delta` | 1e-3 | minimum val MAE (km/h) improvement to reset the patience counter |
| `--seed` | 4711 | seeds Python/NumPy/PyTorch (including weight init) and the train/test/val split |
| `--device` | auto | override auto-detected `cuda`/`mps`/`cpu` |
| `--checkpoint-dir` | `checkpoints` | where checkpoints/history are written |

`train.py` prints the selected device and the train/val/test row counts before
training starts, then one line per epoch with the current LR, and train/val loss and
**MAE**/RMSE in km/h (MAE is the metric that drives early stopping/checkpointing).

**Outputs** (in `--checkpoint-dir`):
- `best.pt` — checkpoint with the lowest validation MAE seen so far (use this for evaluation)
- `last.pt` — checkpoint from the final epoch (equal to `best.pt` only if the last epoch was also the best one)
- `history.json` — per-epoch train/val loss, MAE, and RMSE, for plotting learning curves

Each checkpoint bundles the model weights, the training args (needed to reconstruct
the model architecture and re-derive the same data split), the fitted target
transform (`log_target` flag + the standardization mean/scale, so `evaluate.py`
inverts predictions back to km/h correctly regardless of `--log-target`), and the
fitted `daysUntilHoliday` scaler.

### Trying each activation function

```bash
for act in sigmoid tanh softsign relu leaky_relu; do
    python train.py --data raw-trafic-data.json --activation "$act" \
        --checkpoint-dir "checkpoints-$act"
done
```

See Solution.md's "Activation function comparison" for measured results on the real
export — unlike `MK-3`'s classifier, every activation performs almost identically here.

### Comparing `--log-target` on vs. off

```bash
python train.py --data raw-trafic-data.json --checkpoint-dir /tmp/no-log
python train.py --data raw-trafic-data.json --log-target --checkpoint-dir /tmp/with-log
python evaluate.py --checkpoint /tmp/no-log/best.pt   --data raw-trafic-data.json --output-dir /tmp/no-log/eval
python evaluate.py --checkpoint /tmp/with-log/best.pt --data raw-trafic-data.json --output-dir /tmp/with-log/eval
```

Compare the two runs' printed MAE/RMSE/MAPE/R² — see Solution.md for the measured
trade-off (log-target wins on MAPE, loses slightly on MAE/RMSE).

## Hyperparameter tuning

```bash
python tune.py --data raw-trafic-data.json --trials 20 --trial-epochs 20 --final-epochs 100
```

Random search over `--num-layers`, `--hidden-dim`, `--activation`, `--dropout`, `--lr`,
`--momentum`, `--lr-decay`, `--batch-size`, and `--log-target`, seeded by
`--search-seed` (default 4711 — see `Solution.md`). Every trial reuses `train.py`'s
early-stopping-aware training loop; the winner (by best validation **MAE**, lower is
better) is retrained for up to `--final-epochs` and checkpointed to
`<output-dir>/best_checkpoint/`.

**Outputs** (in `--output-dir`, default `tuning/`):
- `tuning_results.csv` — every trial's config, validation MAE, and the epoch it stopped at
- `best_config.json` — the winning trial's config
- `best_checkpoint/best.pt` — the retrained winning model, usable directly with `evaluate.py`

Increase `--trials` for a more thorough search once you have GPU time to spend; the
per-trial cost is one early-stopping-bounded `run_training` call, so search cost scales
with `--trials × (average epochs per trial)`, not `--trials × --trial-epochs` directly
(most trials stop well before the cap). See Solution.md's "Hyperparameter tuning"
section for why, on the current data, most trials land within a narrow band of each
other.

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best.pt --data raw-trafic-data.json
```

Prints MAE, RMSE, MAPE, and R² on the held-out 10% test split, and writes:
- `checkpoints/eval/metrics.csv` — the same numbers
- `checkpoints/eval/prediction_plots.png` — a predicted-vs-true scatter plot and a residual histogram

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
    --num-layers 2 --hidden-dim 32 --checkpoint-dir /tmp/mk4-smoke
python evaluate.py --checkpoint /tmp/mk4-smoke/best.pt --data synthetic-trafic-data.json \
    --output-dir /tmp/mk4-smoke/eval
```

This was run as part of building this version (8,000 synthetic rows, plus every
`--activation` choice with `--log-target`) on CPU and MPS, and against the real
`raw-trafic-data.json` export (423,771 rows, on an M2 MacBook's Apple GPU): training
converged (with early stopping) in 9–23 epochs at ~3s/epoch depending on
configuration, and `evaluate.py` produced sensible regression metrics — see
Solution.md's "Measured results" for the actual numbers. `tune.py` was also run
end-to-end against both the synthetic and real data. Delete `/tmp/mk4-smoke` and
`synthetic-trafic-data.json` afterwards; they're scratch output.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `ValueError: raw-trafic-data.json is missing expected fields` | File doesn't match the *revised* `SegmentSpeed` schema — check for the one-hot `holiday_type_*`/`*_status`/`holiday_is_*` fields specifically; a file exported before the backend's schema change will be missing all of them (see "Getting data" above) |
| `json.decoder.JSONDecodeError` on load | File is truncated/invalid JSON — likely built with the old `curl \| jq \| sed` approach; re-export with `export_raw_trafic_data.py` |
| Val MAE barely improves past the first few epochs, regardless of architecture | Expected on the current export — see Solution.md's "Why this task is not really that hard": the `*_status` one-hot inputs already explain most of the variance in `speed`, so the model converges near that ceiling quickly; this is not a bug to chase, see Solution.md's ablation suggestion instead |
| Training stops after only a few epochs | Early stopping fired — check `checkpoints/history.json`; if it stopped too early, raise `--early-stopping-patience` or lower `--early-stopping-min-delta` |
| `CUDA out of memory` / Metal-related OOM on MPS | Lower `--batch-size` (unlikely to matter for this model's size, but possible with a very large `--hidden-dim`) |
| `evaluate.py` metrics look implausible after re-exporting data | `raw-trafic-data.json` changed since training (different sections/date range) — retrain, or point `--data` at the original export |
