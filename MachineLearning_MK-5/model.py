"""Multilayer perceptron (MLP) classifier that predicts the status class (one of
`features.STATUS_VALUES`: freeflow/heavy/congested/impossible) from a single
SegmentSpeed row's engineered feature vector (`features.py::FEATURE_DIM` == 20: the 13
upstream `holiday_type_*`/`holiday_is_*` one-hot fields plus 7 cyclical/scaled
covariates -- see dataset.py).

The `*_status` one-hot fields (`freeflow_status`/`heavy_status`/`congested_status`/
`imposible_status`) are deliberately *not* included among the input features here:
they are this version's target (via `dataset.py::derive_status_class`), not an input --
the reverse of `MachineLearning_MK-4`, which used them as inputs to regress `speed`.
`speed` itself is also excluded as an input, since the status fields are derived from
`speed` upstream and using it would make the classification task close to circular.

There are no categorical embedding tables: `holiday_type_*` and `holiday_is_*` already
arrive as 0/1 one-hot fields from the upstream API, so they're concatenated into the
input vector directly (see features.py::one_hot_features).

The feature vector is fed through `--num-layers` fully-connected hidden layers (width
`--hidden-dim`, activation `--activation`) to a final `nn.Linear` layer producing one
raw logit per status class.

Softmax and status prediction
------------------------------
`forward()` returns raw logits (B, num_classes), *not* softmax-normalized
probabilities. Training uses `nn.CrossEntropyLoss`, which combines `log_softmax` and
negative-log-likelihood internally for numerical stability -- applying `nn.Softmax`
before it would double-apply the normalization and unnecessarily lose precision. The
task's "use a softmax to determine the status value" requirement is implemented at
*inference* time instead, in `predict_status()` below: an explicit `torch.softmax`
turns the logits into class probabilities, and `argmax` over those probabilities picks
the predicted status class. `train.py::run_epoch` and `evaluate.py` both call
`predict_status()` rather than re-deriving this logic.
"""

from __future__ import annotations

import torch
from torch import nn

from features import FEATURE_DIM, STATUS_CARDINALITY

# Command-line-selectable activation functions.
ACTIVATIONS: dict[str, type[nn.Module]] = {
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softsign": nn.Softsign,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
}


def predict_status(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax the raw logits into class probabilities, then argmax to the predicted
    status class. Returns (predicted_class, probabilities) -- both (B,) and (B, C)
    tensors respectively, on whatever device `logits` is on.
    """
    probs = torch.softmax(logits, dim=-1)
    return probs.argmax(dim=-1), probs


class TabularMLPClassifier(nn.Module):
    def __init__(
        self,
        num_layers: int = 2,
        hidden_dim: int = 64,
        activation: str = "relu",
        dropout: float = 0.1,
        input_dim: int = FEATURE_DIM,
        num_classes: int = STATUS_CARDINALITY,
    ):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation {activation!r}; choose from {sorted(ACTIVATIONS)}")
        if num_layers < 0:
            raise ValueError("num_layers must be >= 0 (0 == plain linear/softmax regression)")

        self.activation_name = activation

        act_cls = ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), act_cls(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Random weight initialization (the task's "random initialization of weights"
        paradigm): Kaiming-normal for the ReLU family (matches the nonlinearity's
        actual gain), Xavier-normal for the saturating activations (sigmoid/tanh/
        softsign). Biases start at zero. Reproducible per run via `--seed` (see
        train.py::set_seed), which seeds `torch.manual_seed` before the model is
        constructed.
        """
        if not isinstance(module, nn.Linear):
            return
        if self.activation_name == "relu":
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        elif self.activation_name == "leaky_relu":
            nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
        else:
            nn.init.xavier_normal_(module.weight)
        nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)  # (B, num_classes) raw logits
