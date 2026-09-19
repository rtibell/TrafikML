"""Encoder-decoder (seq2seq) recurrent model for per-segment traffic speed/status
forecasting, with the road segment id folded in as an embedding feature so a single
shared model covers every segment (see README for the per-segment-vs-shared tradeoff).

Setting `--t-out 1` degenerates the decoder to a single step, i.e. a plain LSTM/GRU
one-step-ahead forecaster -- the two architectures named in the task spec are really
the same model at different horizons, so only one implementation is kept.
"""

from __future__ import annotations

import torch
from torch import nn

from dataset import CALENDAR_DIM
from features import HOLIDAY_NUM_CARDINALITY, HOLIDAY_TYPE_CARDINALITY, STATUS_CARDINALITY


def make_rnn(cell: str, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> nn.Module:
    cls = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
    return cls(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout if num_layers > 1 else 0.0,
    )


class CovariateEmbeddings(nn.Module):
    """Shared embedding tables for the categorical covariates, used by both encoder
    and decoder so a "freeflow" or "holidayType=1" means the same vector everywhere.
    """

    def __init__(self, num_segments: int, seg_dim: int, status_dim: int, holiday_type_dim: int, holiday_num_dim: int):
        super().__init__()
        self.segment = nn.Embedding(num_segments, seg_dim)
        self.status = nn.Embedding(STATUS_CARDINALITY, status_dim)
        self.holiday_type = nn.Embedding(HOLIDAY_TYPE_CARDINALITY, holiday_type_dim)
        self.holiday_num = nn.Embedding(HOLIDAY_NUM_CARDINALITY, holiday_num_dim)

    @property
    def step_dim(self) -> int:
        return self.status.embedding_dim + self.holiday_type.embedding_dim + self.holiday_num.embedding_dim

    @property
    def segment_dim(self) -> int:
        return self.segment.embedding_dim


class Seq2SeqTrafficForecaster(nn.Module):
    def __init__(
        self,
        num_segments: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        cell: str = "gru",
        seg_embed_dim: int = 16,
        status_embed_dim: int = 4,
        holiday_type_embed_dim: int = 2,
        holiday_num_embed_dim: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cell = cell
        self.embeddings = CovariateEmbeddings(
            num_segments, seg_embed_dim, status_embed_dim, holiday_type_embed_dim, holiday_num_embed_dim
        )
        step_dim = self.embeddings.step_dim
        seg_dim = self.embeddings.segment_dim

        enc_input_size = 1 + step_dim + CALENDAR_DIM + seg_dim  # speed + status + calendar+holidays + segment
        dec_input_size = 1 + step_dim + CALENDAR_DIM + seg_dim  # prev_speed + prev_status + future calendar + segment

        self.encoder = make_rnn(cell, enc_input_size, hidden_size, num_layers, dropout)
        self.decoder = make_rnn(cell, dec_input_size, hidden_size, num_layers, dropout)

        self.speed_head = nn.Linear(hidden_size, 1)
        self.status_head = nn.Linear(hidden_size, STATUS_CARDINALITY)

    def _encoder_inputs(self, batch) -> torch.Tensor:
        seg = self.embeddings.segment(batch["segment_idx"])  # (B, seg_dim)
        t_in = batch["enc_speed"].shape[1]
        seg_rep = seg.unsqueeze(1).expand(-1, t_in, -1)
        status_emb = self.embeddings.status(batch["enc_status"])
        holiday_type_emb = self.embeddings.holiday_type(batch["enc_holiday_type"])
        holiday_num_emb = self.embeddings.holiday_num(batch["enc_holiday_num"])
        return torch.cat(
            [
                batch["enc_speed"].unsqueeze(-1),
                status_emb,
                holiday_type_emb,
                holiday_num_emb,
                batch["enc_calendar"],
                seg_rep,
            ],
            dim=-1,
        )

    def _decoder_step_input(self, seg, prev_speed_scaled, prev_status, dec_calendar_t, holiday_type_t, holiday_num_t):
        status_emb = self.embeddings.status(prev_status)
        holiday_type_emb = self.embeddings.holiday_type(holiday_type_t)
        holiday_num_emb = self.embeddings.holiday_num(holiday_num_t)
        return torch.cat(
            [prev_speed_scaled.unsqueeze(-1), status_emb, holiday_type_emb, holiday_num_emb, dec_calendar_t, seg],
            dim=-1,
        ).unsqueeze(1)  # (B, 1, dec_input_size)

    def forward(self, batch, target_speed_scaled=None, target_status=None, teacher_forcing_ratio: float = 0.0):
        enc_in = self._encoder_inputs(batch)
        _, hidden = self.encoder(enc_in)

        seg = self.embeddings.segment(batch["segment_idx"])
        t_out = batch["dec_calendar"].shape[1]
        device = enc_in.device
        batch_size = enc_in.shape[0]

        prev_speed_scaled = batch["prev_speed_scaled"]
        prev_status = batch["prev_status"]

        speed_preds = torch.empty(batch_size, t_out, device=device)
        status_logits = torch.empty(batch_size, t_out, STATUS_CARDINALITY, device=device)

        for t in range(t_out):
            step_in = self._decoder_step_input(
                seg,
                prev_speed_scaled,
                prev_status,
                batch["dec_calendar"][:, t, :],
                batch["dec_holiday_type"][:, t],
                batch["dec_holiday_num"][:, t],
            )
            out, hidden = self.decoder(step_in, hidden)
            out = out.squeeze(1)
            speed_t = self.speed_head(out).squeeze(-1)
            status_t = self.status_head(out)

            speed_preds[:, t] = speed_t
            status_logits[:, t, :] = status_t

            use_teacher_forcing = self.training and target_speed_scaled is not None and (
                torch.rand(1).item() < teacher_forcing_ratio
            )
            if use_teacher_forcing:
                prev_speed_scaled = target_speed_scaled[:, t]
                prev_status = target_status[:, t]
            else:
                prev_speed_scaled = speed_t.detach()
                prev_status = status_t.detach().argmax(dim=-1)

        return {"speed_scaled": speed_preds, "status_logits": status_logits}
