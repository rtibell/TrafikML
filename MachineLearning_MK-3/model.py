"""Multilayer perceptron (MLP) classifier that predicts `statusEnum` (one of 4 traffic
status classes -- see `features.STATUS_VALUES`) from a single SegmentSpeed row's
engineered feature vector.

`statusEnum` is deliberately *not* included among the input features here: it is now
the prediction target (previously an embedded input alongside a `speed` regression
target -- see Solution.md for the earlier version of this model). `speed` itself is
also excluded as an input, since `statusEnum` is derived from `speed` upstream and
using it would make the classification task close to circular.

The 7 continuous features (features.py::continuous_features) and 2 categorical
embeddings (holidayType, holidayNum) are concatenated into one fixed-order vector and
fed through `--num-layers` fully-connected hidden layers (width `--hidden-dim`,
activation `--activation`) to a final `nn.Linear` layer producing one raw logit per
status class.

Softmax and statusEnum prediction
----------------------------------
`forward()` returns raw logits (B, num_classes), *not* softmax-normalized
probabilities. Training uses `nn.CrossEntropyLoss`, which combines `log_softmax` and
negative-log-likelihood internally for numerical stability -- applying `nn.Softmax`
before it would double-apply the normalization and unnecessarily lose precision. The
task's "use a softmax to determine statusEnum value" requirement is implemented at
*inference* time instead, in `predict_status()` below: an explicit `torch.softmax`
turns the logits into class probabilities, and `argmax` over those probabilities picks
the predicted `statusEnum`. `train.py::run_epoch` and `evaluate.py` both call
`predict_status()` rather than re-deriving this logic.
"""

from __future__ import annotations

import torch
from torch import nn

from dataset import CONTINUOUS_DIM
from features import HOLIDAY_NUM_CARDINALITY, HOLIDAY_TYPE_CARDINALITY, STATUS_CARDINALITY

# Command-line-selectable activation functions (task's "Instruction" section).
ACTIVATIONS: dict[str, type[nn.Module]] = {
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softsign": nn.Softsign,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
}


def predict_status(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax the raw logits into class probabilities, then argmax to the predicted
    `statusEnum`. Returns (predicted_class, probabilities) -- both (B,) and (B, C)
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
        holiday_type_embed_dim: int = 2,
        holiday_num_embed_dim: int = 4,
        num_classes: int = STATUS_CARDINALITY,
    ):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation {activation!r}; choose from {sorted(ACTIVATIONS)}")
        if num_layers < 0:
            raise ValueError("num_layers must be >= 0 (0 == plain linear/softmax regression)")

        self.activation_name = activation
        self.holiday_type_emb = nn.Embedding(HOLIDAY_TYPE_CARDINALITY, holiday_type_embed_dim)
        self.holiday_num_emb = nn.Embedding(HOLIDAY_NUM_CARDINALITY, holiday_num_embed_dim)

        input_dim = CONTINUOUS_DIM + holiday_type_embed_dim + holiday_num_embed_dim

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

    def forward(self, cont: torch.Tensor, holiday_type: torch.Tensor, holiday_num: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([self.holiday_type_emb(holiday_type), self.holiday_num_emb(holiday_num)], dim=-1)
        x = torch.cat([cont, emb], dim=-1)  # (B, input_dim)
        return self.net(x)                   # (B, num_classes) raw logits
