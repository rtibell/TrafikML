# MachineLearning_MK-4 — Traffic Speed MLP Regressor

A PyTorch multilayer perceptron (MLP) that **regresses `speed`** (km/h) for a single
road-segment measurement, from the revised `SegmentSpeed` feature set: `dayNr`,
`daysUntilHoliday`, `minutesSincDaybreak`, `monthOfYear`, and 17 upstream one-hot
fields (`holiday_type_*`, `*_status`, `holiday_is_*`). See Solution.md for why the
`*_status` fields make this an easier task than it might first appear.

This is the fourth iteration in this repository: `MK-1` was a per-segment LSTM/GRU
sequence forecaster of `speed`, `MK-2` a 1D-CNN regressor of `speed`, `MK-3` an MLP
classifier of `statusEnum`, and this version returns to regressing `speed` with an MLP
over the upstream API's new one-hot feature encoding (no embedding tables needed). See
**[Solution.md](Solution.md)** for the full design rationale.

- **[Solution.md](Solution.md)** — how the model is designed: architecture, data
  contract, the "why this task isn't that hard" analysis, the activation-function and
  `--log-target` experiments, train/test split, measured results, and suggested
  improvements.
- **[Operations.md](Operations.md)** — how to set up, get data, train, tune, and
  evaluate, plus a troubleshooting table.

```bash
cd MachineLearning_MK-4
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python export_raw_trafic_data.py --base-url http://localhost:8080 --out raw-trafic-data.json
python train.py --data raw-trafic-data.json --epochs 60 --num-layers 2 --activation relu
python evaluate.py --checkpoint checkpoints/best.pt --data raw-trafic-data.json
```

## Command-line surface (task requirements)

| Requirement | Flag |
|---|---|
| Number of layers in the MLP | `--num-layers` |
| Activation function | `--activation {sigmoid,tanh,softsign,relu,leaky_relu}` |
| Log-transform the target or not | `--log-target` |

## ML paradigms implemented

Gradient descent (`torch.optim.SGD`) and backpropagation, minibatch training
(`DataLoader`), early stopping (on validation MAE), adjustable learning rate and
momentum (with optional per-epoch LR decay), and random weight initialization matched
to the chosen activation. See Solution.md's "Training paradigms" table for exactly
where each one lives in the code.

## Input normalization

`daysUntilHoliday` is standardized; `dayNr`/`monthOfYear`/`minutesSincDaybreak` are
cyclically (sin/cos) encoded; the 17 upstream one-hot fields are passed through
unchanged. `speed` (the target) is optionally `log1p`-transformed (`--log-target`) and
always standardized, both fit on the training split only to avoid test-set leakage.

## Device support

Auto-detects, in order: CUDA (e.g. a GTX 1080) → Apple GPU via Metal/MPS (e.g. an M2
MacBook) → CPU. Override with `--device {cuda,mps,cpu}`.
