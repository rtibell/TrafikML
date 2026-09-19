# MachineLearning_MK-3 — Traffic Status MLP Classifier

A PyTorch multilayer perceptron (MLP) that **classifies** `statusEnum` — one of four
traffic-status classes (`freeflow=0, heavy=1, congested=2, impossible=3`) — for a
single road-segment measurement, from time-of-day and calendar covariates alone
(`speed` and `statusEnum` itself are deliberately excluded from the inputs; see
Solution.md for why).

This is the third iteration in this repository, and the second revision of this
directory: `MK-1` was a per-segment LSTM/GRU sequence forecaster of `speed`, `MK-2` a
1D-CNN regressor of `speed`, this directory's first version an MLP regressor of
`speed`, and this version retargets the same MLP to classify `statusEnum` with a
softmax output. See **[Solution.md](Solution.md)** for the full design rationale.

- **[Solution.md](Solution.md)** — how the model is designed: architecture, data
  contract, class-balance analysis, the softmax/loss-weighting design decisions, the
  activation-function experiment, train/test split, measured results, and suggested
  improvements.
- **[Operations.md](Operations.md)** — how to set up, get data, train, tune, and
  evaluate, plus a troubleshooting table.

```bash
cd MachineLearning_MK-3
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

## ML paradigms implemented

Gradient descent (`torch.optim.SGD`) and backpropagation, minibatch training
(`DataLoader`), early stopping (on validation *balanced* accuracy — see Solution.md),
adjustable learning rate and momentum (with optional per-epoch LR decay), and random
weight initialization matched to the chosen activation. See Solution.md's "Training
paradigms" table for exactly where each one lives in the code.

## Softmax and class imbalance

The model outputs raw logits; `model.py::predict_status` applies `torch.softmax` then
`argmax` to produce the predicted `statusEnum`, exactly as the task specifies. Training
uses `nn.CrossEntropyLoss` (numerically equivalent, but combines `log_softmax` + NLL
for stability). `statusEnum` is heavily imbalanced in the real data (87.7% freeflow,
0.27% impossible) — class-weighted loss and *balanced* accuracy (not raw accuracy)
drive training and evaluation by default; see Solution.md for the measured effect of
getting this wrong.

## Device support

Auto-detects, in order: CUDA (e.g. a GTX 1080) → Apple GPU via Metal/MPS (e.g. an M2
MacBook) → CPU. Override with `--device {cuda,mps,cpu}`.
