"""Multilayer perceptron (MLP) that regresses `speed` (km/h) from a single SegmentSpeed
row's engineered feature vector (`features.py::FEATURE_DIM` == 24: the 17 upstream
one-hot fields plus 7 cyclical/scaled covariates -- see dataset.py).

Unlike earlier versions of this model (MK-2's CNN, MK-3's classifier), there are no
categorical embedding tables here: `holiday_type_*`, `*_status`, and `holiday_is_*`
already arrive as 0/1 one-hot fields from the upstream API, so they're concatenated
into the input vector directly rather than looked up through an `nn.Embedding`.

The feature vector is fed through `--num-layers` fully-connected hidden layers (width
`--hidden-dim`, activation `--activation`) to a final `nn.Linear(hidden_dim, 1)` layer
producing one scalar (scaled) speed prediction per row.
"""

from __future__ import annotations

import torch
from torch import nn

from features import FEATURE_DIM

# Command-line-selectable activation functions (task's "Instruction" section).
ACTIVATIONS: dict[str, type[nn.Module]] = {
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softsign": nn.Softsign,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
}


class TabularMLPRegressor(nn.Module):
    def __init__(
        self,
        num_layers: int = 2,
        hidden_dim: int = 64,
        activation: str = "relu",
        dropout: float = 0.1,
        input_dim: int = FEATURE_DIM,
    ):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation {activation!r}; choose from {sorted(ACTIVATIONS)}")
        if num_layers < 0:
            raise ValueError("num_layers must be >= 0 (0 == plain linear regression)")

        self.activation_name = activation

        act_cls = ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), act_cls(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
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
        return self.net(features).squeeze(-1)  # (B,) scaled speed prediction
