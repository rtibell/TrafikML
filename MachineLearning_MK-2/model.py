"""1D-CNN regressor that predicts `speed` (km/h) from a single SegmentSpeed row's
engineered feature vector, per the task's explicit request to "base the model on a
CNN." See Solution.md ("Why a CNN for row-wise tabular data") for the design
rationale and its tradeoffs against the more usual MLP for this kind of input.

The 7 continuous features (features.py::continuous_features) and the 3 categorical
embeddings (statusEnum, holidayType, holidayNum) are concatenated into one fixed-order
vector, then treated as a length-L, 1-channel signal so `nn.Conv1d` layers can slide
small kernels across neighbouring feature slots and mix them before pooling down to a
single vector for the regression head.
"""

from __future__ import annotations

import torch
from torch import nn

from dataset import CONTINUOUS_DIM
from features import HOLIDAY_NUM_CARDINALITY, HOLIDAY_TYPE_CARDINALITY, STATUS_CARDINALITY


class TabularCNNRegressor(nn.Module):
    def __init__(
        self,
        conv_channels: tuple[int, ...] = (32, 64),
        kernel_size: int = 3,
        dropout: float = 0.1,
        status_embed_dim: int = 4,
        holiday_type_embed_dim: int = 2,
        holiday_num_embed_dim: int = 4,
    ):
        super().__init__()
        self.status_emb = nn.Embedding(STATUS_CARDINALITY, status_embed_dim)
        self.holiday_type_emb = nn.Embedding(HOLIDAY_TYPE_CARDINALITY, holiday_type_embed_dim)
        self.holiday_num_emb = nn.Embedding(HOLIDAY_NUM_CARDINALITY, holiday_num_embed_dim)

        self.input_length = CONTINUOUS_DIM + status_embed_dim + holiday_type_embed_dim + holiday_num_embed_dim

        conv_layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in conv_channels:
            conv_layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_channels = out_channels
        self.conv = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        head_hidden = max(in_channels // 2, 8)
        self.head = nn.Sequential(
            nn.Linear(in_channels, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, cont: torch.Tensor, status: torch.Tensor, holiday_type: torch.Tensor, holiday_num: torch.Tensor) -> torch.Tensor:
        emb = torch.cat(
            [self.status_emb(status), self.holiday_type_emb(holiday_type), self.holiday_num_emb(holiday_num)],
            dim=-1,
        )
        x = torch.cat([cont, emb], dim=-1)          # (B, input_length)
        x = x.unsqueeze(1)                          # (B, 1, input_length) -- 1 "channel", input_length-long signal
        x = self.conv(x)                             # (B, C, input_length)
        x = self.pool(x).squeeze(-1)                 # (B, C)
        return self.head(x).squeeze(-1)               # (B,) scaled speed prediction
