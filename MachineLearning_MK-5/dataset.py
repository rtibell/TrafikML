"""Loading raw-trafic-data.json, normalizing features, and the 90/10 train/test split
(shuffled, seed 4711 per the task spec).

The prediction target is the 4-class status one-hot group (`freeflow_status`,
`heavy_status`, `congested_status`, `imposible_status` -- see
`derive_status_class` below), classified from the non-circular `SegmentSpeed`
covariates: `dayNr`, `daysUntilHoliday`, `minutesSincDaybreak`, `monthOfYear`, and the
13 `holiday_type_*`/`holiday_is_*` one-hot fields. `speed` is deliberately not used as
an input -- the status fields are derived from `speed` upstream by the collector (see
SpeedOfSectionMLData.java's status thresholds), so including it would make the
classification task close to circular. See Solution.md for the measured class balance
and why class-weighted loss is used by default.

Every SegmentSpeed record is treated as an independent, unordered row: there is no
`sectionId` and no `measureTime` ordering in the feature set, so there is nothing to
window over.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from features import (
    CONTINUOUS_FEATURE_DIM,
    HOLIDAY_IS_FIELDS,
    HOLIDAY_TYPE_FIELDS,
    STATUS_CARDINALITY,
    STATUS_FIELDS,
    continuous_features,
    one_hot_features,
)

RANDOM_SEED = 4711

REQUIRED_FIELDS = (
    {"dayNr", "daysUntilHoliday", "minutesSincDaybreak", "monthOfYear"}
    | set(HOLIDAY_TYPE_FIELDS)
    | set(HOLIDAY_IS_FIELDS)
    | set(STATUS_FIELDS)
)


def load_dataframe(json_path: str) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame.from_records(records)
    missing = REQUIRED_FIELDS - set(df.columns)
    if missing:
        raise ValueError(f"{json_path} is missing expected fields: {sorted(missing)}")
    df = df.dropna(subset=sorted(REQUIRED_FIELDS)).reset_index(drop=True)
    return df


def derive_status_class(df: pd.DataFrame) -> np.ndarray:
    """The task's target variable: argmax over the four `*_status` one-hot columns,
    in `features.STATUS_FIELDS`/`STATUS_VALUES` order (freeflow=0, heavy=1,
    congested=2, impossible=3). Using argmax over the one-hot fields directly (rather
    than reading the redundant `statusEnum` column) is deliberate -- it's exactly what
    the task specifies as the target, and is robust even if a future export drops
    `statusEnum` but keeps the one-hot fields.
    """
    return df[STATUS_FIELDS].to_numpy(dtype=np.float32).argmax(axis=1).astype(np.int64)


def split_train_test(indices: np.ndarray, test_size: float = 0.1, seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Split an array of row indices. `shuffle=True` (the default) performs the
    task-mandated random shuffle; `random_state=seed` makes it reproducible. Called
    twice in train.py: once for the 90/10 train/test split, once more on the 90% train
    partition to carve out a validation slice for early stopping / model selection.
    """
    return train_test_split(indices, test_size=test_size, random_state=seed, shuffle=True)


class FixedScaler:
    """Minimal drop-in for sklearn's StandardScaler.transform(), reconstructed from
    the mean/scale saved in a training checkpoint so evaluation doesn't need to refit
    (and can't silently drift from the scaler used at train time).
    """

    def __init__(self, mean: list[float], scale: list[float]):
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.scale_ = np.asarray(scale, dtype=np.float64)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.scale_


def fit_scalers(df: pd.DataFrame, train_idx: np.ndarray) -> dict:
    """Fit the `daysUntilHoliday` feature scaler on the *training* rows only, to avoid
    test-set leakage into the normalization statistics -- the task's "normalize input
    data" requirement.
    """
    train_rows = df.iloc[train_idx]
    days_scaler = StandardScaler().fit(train_rows["daysUntilHoliday"].to_numpy(dtype=np.float64).reshape(-1, 1))
    return {"days_until_holiday": days_scaler}


def compute_class_weights(df: pd.DataFrame, train_idx: np.ndarray, num_classes: int = STATUS_CARDINALITY) -> torch.Tensor:
    """`sklearn`-style "balanced" class weights: `n_samples / (n_classes * count[c])`,
    computed on the *training* rows only. The status target is heavily imbalanced in
    the real export (~86% freeflow, ~0.3% impossible -- see Solution.md), so
    unweighted cross-entropy would let the model ignore the rare-but-important
    congested/impossible classes almost entirely.
    """
    counts = np.bincount(derive_status_class(df.iloc[train_idx]), minlength=num_classes)
    counts = np.clip(counts, 1, None)  # avoid div-by-zero if a class is absent from this split
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


class SegmentSpeedDataset(Dataset):
    """Each sample is one independent SegmentSpeed row: a single 20-wide engineered
    feature vector (13 upstream one-hot fields ++ 7 cyclical/scaled covariates -- see
    features.py) -> status class (0..3).
    """

    def __init__(self, df: pd.DataFrame, indices: np.ndarray, scalers: dict):
        rows = df.iloc[indices]
        cont = continuous_features(
            rows["dayNr"].to_numpy(),
            rows["monthOfYear"].to_numpy(),
            rows["minutesSincDaybreak"].to_numpy(),
            rows["daysUntilHoliday"].to_numpy(),
            scalers["days_until_holiday"],
        )
        one_hot = one_hot_features(rows)
        self.features = torch.from_numpy(np.concatenate([one_hot, cont], axis=1))
        self.status = torch.from_numpy(derive_status_class(rows))

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "features": self.features[idx],
            "status": self.status[idx],
        }


assert CONTINUOUS_FEATURE_DIM == 7
