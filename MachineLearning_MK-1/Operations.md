# Operations — running MachineLearning_MK-1

How to set up, get data, train, and evaluate the traffic forecaster. See `Solution.md`
for design rationale.

## Setup

```bash
cd MachineLearning_MK-1
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

CUDA (GTX 1080) is picked up automatically if `torch.cuda.is_available()` — no code
change needed; `train.py`/`evaluate.py` print which device they selected. The 1080 is
a Pascal card (no tensor cores), so `--amp` mixed precision mainly saves VRAM rather
than speeding things up materially; it's opt-in, off by default.

## Getting data

The pipeline expects `raw-trafic-data.json` in this directory: a JSON array of
`SegmentSpeed` records (see `Solution.md`'s Data contract section for the exact
fields).

### Option A — export from a running backend (recommended)

```bash
# with the Spring Boot backend running (e.g. ./gradlew bootRun --args='--spring.profiles.active=dev')
python export_raw_trafic_data.py --base-url http://localhost:8080 --out raw-trafic-data.json
```

This is stdlib-only (no venv needed to run it) and walks every section from
`GET /api/v1/sections`, pulls each one's `/ml-speed-data`, and writes a single valid
JSON array.

> **Do not** use the `curl ... | jq | sed 's/]/,/' >> raw-trafic-data.json` pattern
> from `gen_raw-trafic-data.sh` at the repo root — it only fetches one section per
> invocation and the `sed` trick does not produce valid, closed JSON. If
> `raw-trafic-data.json` was built that way, re-export it with the script above
> before training; `load_dataframe()` in `dataset.py` will raise a JSON decode error
> on a truncated file.

### Option B — synthetic data (smoke test, no backend needed)

```bash
python generate_synthetic_data.py --out synthetic-trafic-data.json --num-segments 20 --days 14
```

Fabricates a rough rush-hour speed profile across N synthetic segments. Useful to
confirm the pipeline runs (dataset → model → train → evaluate) before pointing it at
real data, or to develop offline without a running backend. Writes to
`synthetic-trafic-data.json` by default — **never** point it at `raw-trafic-data.json`.
Its holiday/calendar fields are simplified stand-ins, not the real Swedish holiday
calendar `SwedishHolidays.java` computes — don't use it to evaluate holiday-related
model behavior.

## Training

```bash
python train.py --data raw-trafic-data.json --epochs 30 --batch-size 256
```

Key flags (all optional, see `python train.py --help` for the full list):

| Flag | Default | Meaning |
|---|---|---|
| `--t-in` | 60 | minutes of history fed to the encoder |
| `--t-out` | 15 | minutes ahead to forecast |
| `--stride` | 5 | step between consecutive windows from the same segment |
| `--val-fraction` | 0.1 | held out from the 90% train split, for early-stopping monitoring only |
| `--hidden-size` | 128 | RNN hidden size |
| `--num-layers` | 2 | RNN depth |
| `--cell` | gru | `gru` or `lstm` |
| `--status-loss-weight` | 0.3 | weight of the status cross-entropy term vs. speed MSE |
| `--teacher-forcing-ratio` | 0.5 | probability of feeding true (not predicted) previous speed/status to the decoder during training |
| `--tf-decay-to-zero` | off | linearly decay teacher forcing to 0 over `--epochs` |
| `--seed` | 4711 | seeds Python/NumPy/PyTorch and the train/test split |
| `--device` | auto | override auto-detected `cuda`/`cpu` |
| `--amp` | off | mixed precision (CUDA only) |
| `--checkpoint-dir` | `checkpoints` | where checkpoints/history are written |

`train.py` prints the selected device, the number of windows built, and the
train/val/test split sizes before training starts, then one line per epoch with
train/val loss (and the speed/status sub-losses). Watch for val loss diverging from
train loss (overfitting) or plateauing early (may need a lower `--lr`, more capacity,
or more data).

**Outputs** (in `--checkpoint-dir`):
- `best.pt` — checkpoint with the lowest validation loss seen so far (use this for evaluation)
- `last.pt` — checkpoint from the final epoch
- `history.json` — per-epoch train/val loss breakdown, for plotting learning curves

Each checkpoint bundles the model weights, the training args (needed to reconstruct
the model architecture and re-derive the same data split), the segment-id-to-embedding
index mapping, and the fitted feature scalers — `evaluate.py` needs nothing else.

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best.pt --data raw-trafic-data.json
```

Prints per-forecast-horizon-minute metrics (speed MAE/RMSE/MAPE in km/h, status
accuracy and macro-F1), plus overall pooled numbers, and writes:
- `checkpoints/eval/metrics.csv` — the same per-horizon table
- `checkpoints/eval/horizon_metrics.png` — MAE and status accuracy vs. forecast horizon

**Important**: the test split is *recomputed*, not stored — `evaluate.py` rebuilds
windows from `--data` (or the path stored in the checkpoint's training args, if
`--data` is omitted) and re-runs the same seeded shuffle/split. This only reproduces
the original held-out test set if `raw-trafic-data.json` and the `t_in`/`t_out`/
`stride` hyperparameters are unchanged since training. If you've re-exported or
appended to `raw-trafic-data.json` since training a checkpoint, evaluate against a
copy of the file as it was at training time, or retrain.

If segment ids in the current data don't match those the checkpoint was trained on,
`evaluate.py` prints a warning — this usually means `raw-trafic-data.json` has been
regenerated with a different set of sections and the embedding lookup will be
misaligned; retrain instead of trusting those numbers.

## End-to-end smoke test

A quick way to confirm everything is wired correctly before spending real GPU time:

```bash
python generate_synthetic_data.py --out synthetic-trafic-data.json --num-segments 8 --days 6
python train.py --data synthetic-trafic-data.json --epochs 8 --batch-size 64 \
    --t-in 30 --t-out 10 --stride 15 --hidden-size 32 --num-layers 1 \
    --checkpoint-dir /tmp/mk1-smoke
python evaluate.py --checkpoint /tmp/mk1-smoke/best.pt --data synthetic-trafic-data.json \
    --output-dir /tmp/mk1-smoke/eval
```

This was run on CPU (small hidden size, few epochs) as part of building this project:
train/val loss decreased every epoch with no shape or data-leakage errors, and
evaluation produced sensible, roughly horizon-flat metrics (~2.5 km/h speed MAE,
~89–90% status accuracy on the synthetic rush-hour profile) — confirming the pipeline
mechanics work, not that these numbers mean anything for real trafiken.nu data. Delete
`/tmp/mk1-smoke` and `synthetic-trafic-data.json` afterwards; they're scratch output.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `ValueError: raw-trafic-data.json is missing expected fields` | File doesn't match the `SegmentSpeed` schema — check field names, especially `holidayType` vs. the API's `holidayNr` |
| `json.decoder.JSONDecodeError` on load | File is truncated/invalid JSON — likely built with the old `curl \| jq \| sed` approach; re-export with `export_raw_trafic_data.py` |
| `Built 0 windows across N segments` | No segment has `t_in + t_out` consecutive gap-free minutes — check `--t-in`/`--t-out` against actual data density, or data has more gaps than expected |
| Val loss much worse than train loss | Overfitting — reduce `--hidden-size`/`--num-layers`, add more data, or increase `--dropout` |
| `CUDA out of memory` | Lower `--batch-size` or `--hidden-size`, or drop `--amp` (it lowers memory but adds overhead if VRAM isn't actually the bottleneck) |
| `evaluate.py` warns about segment id mismatch | `raw-trafic-data.json` changed since training (different sections present) — retrain, or point `--data` at the original export |
