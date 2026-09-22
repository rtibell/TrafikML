# MachineLearning_MK-5 — Traffic Status MLP Classifier

A PyTorch multilayer perceptron (MLP) that **classifies** the traffic status of a
single road-segment measurement — `freeflow_status`, `heavy_status`,
`congested_status`, or `imposible_status` — from time-of-day and calendar covariates
alone (`speed` and the status fields themselves are deliberately excluded from the
inputs; see Solution.md for why).

This is the fifth iteration in this repository: `MK-1` was a per-segment LSTM/GRU
sequence forecaster of `speed`, `MK-2` a 1D-CNN regressor of `speed`, `MK-3` an MLP
classifier of `statusEnum`, `MK-4` an MLP regressor of `speed` using the status fields
as *inputs*, and this version classifies the status fields again — now expressed as
the revised schema's `*_status` one-hot group rather than `MK-3`'s integer `statusEnum`
— with a softmax output. See **[Solution.md](Solution.md)** for the full design
rationale, including why a single softmax head (not four independent sigmoids) is the
correct fit for this target.

- **[Solution.md](Solution.md)** — how the model is designed: architecture, data
  contract, the softmax-vs-independent-sigmoids design decision, class-balance
  analysis, the activation-function experiment, train/test split, measured results,
  and suggested improvements.
- **[Operations.md](Operations.md)** — how to set up, get data, train, tune, and
  evaluate, plus a troubleshooting table.
- **[Tuning.md](Tuning.md)** — how hyperparameter tuning works: the random-search
  (`tune-rnd.py`) and genetic-algorithm (`tune-ga.py`) tuners, why PyGAD was chosen, a
  measured comparison between the two, and tips for tuning the tuner.

```bash
cd MachineLearning_MK-5
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python export_raw_trafic_data.py --base-url http://localhost:8080 --out raw-trafic-data.json
python train.py --data raw-trafic-data.json --epochs 60 --num-layers 2 --activation relu
python evaluate.py --checkpoint checkpoints/best.pt --data raw-trafic-data.json
```

## Command-line surface

| Requirement | Flag |
|---|---|
| Number of layers in the MLP | `--num-layers` |
| Activation function | `--activation {sigmoid,tanh,softsign,relu,leaky_relu}` |
| Class-weighted loss | `--class-weights {balanced,none}` |

## ML paradigms implemented

Gradient descent (`torch.optim.SGD`) and backpropagation, minibatch training
(`DataLoader`), early stopping (on validation *balanced* accuracy — see Solution.md),
adjustable learning rate and momentum (with optional per-epoch LR decay), and random
weight initialization matched to the chosen activation. See Solution.md's "Training
paradigms" table for exactly where each one lives in the code.

## Softmax, not four independent sigmoids

`freeflow_status`/`heavy_status`/`congested_status`/`imposible_status` are a one-hot
encoding of a single 4-way categorical variable (exactly one is ever 1), not four
independent binary labels. `dataset.py::derive_status_class` collapses them to one
class index; the model outputs raw logits over the 4 classes, and
`model.py::predict_status` applies `torch.softmax` then `argmax` to produce the
predicted status — the same single-label multi-class framing `MK-3` used, just against
the revised one-hot target fields. Training uses `nn.CrossEntropyLoss` (numerically
equivalent, but combines `log_softmax` + NLL for stability). The target is heavily
imbalanced in the real data (86.9% freeflow, 0.25% impossible) — class-weighted loss
and *balanced* accuracy (not raw accuracy) drive training and evaluation by default;
see Solution.md for the measured effect of getting this wrong.

## Device support

Auto-detects, in order: CUDA (e.g. a GTX 1080) → Apple GPU via Metal/MPS (e.g. an M2
MacBook) → CPU. Override with `--device {cuda,mps,cpu}`.
