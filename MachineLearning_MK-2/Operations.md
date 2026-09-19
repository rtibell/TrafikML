# Operations — running MachineLearning_MK-2

How to set up, get data, train, tune, and evaluate the CNN speed regressor. See
`Solution.md` for design rationale and measured results.

## Setup

```bash
cd MachineLearning_MK-2
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

CUDA (GTX 1080) is picked up automatically if `torch.cuda.is_available()` — no code
change needed; `train.py`/`tune.py`/`evaluate.py` print which device they selected.
The 1080 is a Pascal card (no tensor cores); this model is small enough (a few conv
layers over a 17-wide input) that it trains in seconds per epoch even on CPU, so GPU
mainly helps by letting `--batch-size` and `tune.py`'s trial count scale up cheaply.

## Getting data

The pipeline expects `raw-trafic-data.json` in this directory: a JSON array of
`SegmentSpeed` records (see `Solution.md`'s Data contract section for the exact
fields).

### Option A — export from a running backend (recommended)

```bash
# with the Spring Boot backend running (e.g. ./gradlew bootRun --args='--spring.profiles.active=dev')
python export_raw_trafic_data.py --base-url http://localhost:8080 --out raw-trafic-data.json
```

Stdlib-only (no venv needed to run it); walks every section from
`GET /api/v1/sections`, pulls each one's `/ml-speed-data`, and writes a single valid
JSON array. Identical to `MachineLearning_MK-1`'s script of the same name.

> **Do not** use the `curl ... | jq | sed 's/]/,/' >> raw-trafic-data.json` pattern
> from `gen_raw-trafic-data.sh` at the repo root — it only fetches one section per
> invocation and the `sed` trick does not produce valid, closed JSON.

### Option B — synthetic data (smoke test, no backend needed)

```bash
python generate_synthetic_data.py --out synthetic-trafic-data.json --rows 20000
```

Fabricates independent rows with a rough rush-hour speed profile (no per-segment
continuity is needed here — see Solution.md on why this model treats rows as i.i.d.).
Useful to confirm the pipeline runs (dataset → model → train → tune → evaluate) before
pointing it at real data. Writes to `synthetic-trafic-data.json` by default — **never**
point it at `raw-trafic-data.json`. Its holiday/calendar fields are simplified
stand-ins, not the real Swedish holiday calendar `SwedishHolidays.java` computes.

## Training

```bash
python train.py --data raw-trafic-data.json --epochs 30 --batch-size 512
```

Key flags (all optional; `python train.py --help` for the full list):

| Flag | Default | Meaning |
|---|---|---|
| `--val-fraction` | 0.1 | held out from the 90% train split, for model selection only |
| `--batch-size` | 512 | |
| `--conv-channels` | `32,64` | comma-separated `Conv1d` output channels, e.g. `32,64,64` |
| `--kernel-size` | 3 | |
| `--dropout` | 0.1 | |
| `--status-embed-dim` / `--holiday-type-embed-dim` / `--holiday-num-embed-dim` | 4 / 2 / 4 | categorical embedding sizes |
| `--log-target` | off | train on `log1p(speed)` instead of raw `speed` — see Solution.md; measured slightly worse on real data, so off by default |
| `--lr` | 1e-3 | |
| `--seed` | 4711 | seeds Python/NumPy/PyTorch and the train/test/val split |
| `--device` | auto | override auto-detected `cuda`/`cpu` |
| `--checkpoint-dir` | `checkpoints` | where checkpoints/history are written |

`train.py` prints the selected device and the train/val/test row counts before
training starts, then one line per epoch with train/val loss and MAE (km/h). Watch for
val MAE diverging from train MAE (overfitting) or plateauing early (try a lower `--lr`,
more/fewer conv channels, or more data).

**Outputs** (in `--checkpoint-dir`):
- `best.pt` — checkpoint with the lowest validation MAE seen so far (use this for evaluation)
- `last.pt` — checkpoint from the final epoch
- `history.json` — per-epoch train/val loss and MAE, for plotting learning curves

Each checkpoint bundles the model weights, the training args (needed to reconstruct
the model architecture and re-derive the same data split), the fitted `daysUntilHoliday`
scaler, and the fitted target transform (mean/scale, and whether `--log-target` was
used) — `evaluate.py` needs nothing else.

## Hyperparameter tuning

```bash
python tune.py --data raw-trafic-data.json --trials 20 --trial-epochs 8 --final-epochs 30
```

Random search over architecture/optimization hyperparameters and `--log-target`,
seeded by `--search-seed` (default 4711 — see `Solution.md`). Each trial trains for
`--trial-epochs` (no checkpoint written) and is ranked by validation MAE; the winner is
retrained for `--final-epochs` and checkpointed to `<output-dir>/best_checkpoint/`.

**Outputs** (in `--output-dir`, default `tuning/`):
- `tuning_results.csv` — every trial's config and validation MAE
- `best_config.json` — the winning trial's config
- `best_checkpoint/best.pt` — the retrained winning model, usable directly with `evaluate.py`

Increase `--trials` for a more thorough search once you have GPU time to spend; the
per-trial cost is one short `run_training` call, so search cost scales linearly with
`--trials × --trial-epochs`.

### Comparing raw vs. log target directly

To reproduce the Solution.md comparison without a full random search:

```bash
python train.py --data raw-trafic-data.json --epochs 15 --checkpoint-dir /tmp/nolog
python train.py --data raw-trafic-data.json --epochs 15 --log-target --checkpoint-dir /tmp/log
python evaluate.py --checkpoint /tmp/nolog/best.pt --data raw-trafic-data.json --output-dir /tmp/nolog/eval
python evaluate.py --checkpoint /tmp/log/best.pt   --data raw-trafic-data.json --output-dir /tmp/log/eval
```

Compare the two `metrics.csv` files (or the printed MAE/RMSE/R² lines) directly.

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best.pt --data raw-trafic-data.json
```

Prints MAE, RMSE, MAPE, and R² (km/h) on the held-out 10% test split, and writes:
- `checkpoints/eval/metrics.csv` — the same numbers
- `checkpoints/eval/prediction_plots.png` — predicted-vs-true scatter and a residual histogram

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
python train.py --data synthetic-trafic-data.json --epochs 5 --batch-size 256 \
    --conv-channels 16,32 --checkpoint-dir /tmp/mk2-smoke
python evaluate.py --checkpoint /tmp/mk2-smoke/best.pt --data synthetic-trafic-data.json \
    --output-dir /tmp/mk2-smoke/eval
```

This was run on CPU as part of building this project against the real
`raw-trafic-data.json` export (107,953 rows): training and evaluation both completed
without shape or data-leakage errors, ~9–10s/epoch, and `evaluate.py` produced sensible
metrics (see Solution.md's "Measured results" for the actual numbers) — confirming the
pipeline mechanics work end to end on real data. Delete `/tmp/mk2-smoke` and
`synthetic-trafic-data.json` afterwards; they're scratch output.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `ValueError: raw-trafic-data.json is missing expected fields` | File doesn't match the `SegmentSpeed` schema — check field names, especially `holidayType` vs. the API's `holidayNr` |
| `json.decoder.JSONDecodeError` on load | File is truncated/invalid JSON — likely built with the old `curl \| jq \| sed` approach; re-export with `export_raw_trafic_data.py` |
| Val MAE much worse than train MAE | Overfitting — reduce `--conv-channels`/embedding dims, add more data, or increase `--dropout` |
| Val MAE barely moves across epochs | Learning rate too low/high, or the model has already converged — check `checkpoints/history.json`; this dataset's val MAE typically plateaus by epoch ~10 |
| `CUDA out of memory` | Lower `--batch-size` (unlikely to matter for this model's size, but possible with a very large `--conv-channels` list) |
| `evaluate.py` metrics look implausible after re-exporting data | `raw-trafic-data.json` changed since training (different sections/date range) — retrain, or point `--data` at the original export |
