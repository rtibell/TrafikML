"""Loading raw-trafic-data.json, normalizing features/target, and the 90/10 train/test
split (shuffled, seed 4711 per the task spec).

Unlike MachineLearning_MK-1 (a per-segment sliding-window sequence model), this
version treats every SegmentSpeed record as an independent, unordered row: the task
spec's feature list (statusEnum, dayNr, daysUntilHoliday, holidayType,
minutesSincDaybreak, monthOfYear, holidayNum) carries no `sectionId` and no
`measureTime` ordering, so there is nothing to window over -- see Solution.md for the
consequences of that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from features import CONTINUOUS_FEATURE_DIM, continuous_features

RANDOM_SEED = 4711

REQUIRED_FIELDS = {
    "statusEnum", "speed", "dayNr", "daysUntilHoliday",
    "holidayType", "minutesSincDaybreak", "monthOfYear", "holidayNum",
}


def load_dataframe(json_path: str) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame.from_records(records)
    missing = REQUIRED_FIELDS - set(df.columns)
    if missing:
        raise ValueError(f"{json_path} is missing expected fields: {sorted(missing)}")
    df = df.dropna(subset=sorted(REQUIRED_FIELDS)).reset_index(drop=True)
    return df


def split_train_test(indices: np.ndarray, test_size: float = 0.1, seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Split an array of row indices. `shuffle=True` (the default) performs the
    task-mandated random shuffle; `random_state=seed` makes it reproducible. Called
    twice in train.py: once for the 90/10 train/test split, once more on the 90% train
    partition to carve out a validation slice for early stopping / model selection.
    """
    return train_test_split(indices, test_size=test_size, random_state=seed, shuffle=True)


class FixedScaler:
    """Minimal drop-in for sklearn's StandardScaler.transform()/inverse_transform(),
    reconstructed from the mean/scale saved in a training checkpoint so evaluation
    doesn't need to refit (and can't silently drift from the scaler used at train time).
    """

    def __init__(self, mean: list[float], scale: list[float]):
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.scale_ = np.asarray(scale, dtype=np.float64)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.scale_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale_ + self.mean_


@dataclass
class TargetTransform:
    """speed (km/h) -> optionally log1p -> standardized, and back.

    `log_target` is the switch the task asks us to evaluate ("try out if target
    variable should be transformed with log or not") -- see Solution.md for the
    measured outcome on real data.
    """

    log_target: bool
    scaler: StandardScaler | FixedScaler

    @classmethod
    def fit(cls, speed_train: np.ndarray, log_target: bool) -> "TargetTransform":
        y = np.log1p(speed_train) if log_target else speed_train
        scaler = StandardScaler().fit(np.asarray(y, dtype=np.float64).reshape(-1, 1))
        return cls(log_target=log_target, scaler=scaler)

    @classmethod
    def from_checkpoint(cls, log_target: bool, mean: list[float], scale: list[float]) -> "TargetTransform":
        return cls(log_target=log_target, scaler=FixedScaler(mean, scale))

    def transform(self, speed: np.ndarray) -> np.ndarray:
        y = np.log1p(speed) if self.log_target else speed
        return self.scaler.transform(np.asarray(y, dtype=np.float64).reshape(-1, 1)).astype(np.float32).reshape(-1)

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        y = self.scaler.inverse_transform(np.asarray(y_scaled, dtype=np.float64).reshape(-1, 1)).reshape(-1)
        return np.expm1(y) if self.log_target else y

    def state_dict(self) -> dict:
        return {"log_target": self.log_target, "mean": self.scaler.mean_.tolist(), "scale": self.scaler.scale_.tolist()}


def fit_scalers(df: pd.DataFrame, train_idx: np.ndarray, log_target: bool) -> dict:
    """Fit both the feature scaler (daysUntilHoliday) and the target transform on the
    *training* rows only, to avoid test-set leakage.
    """
    train_rows = df.iloc[train_idx]
    days_scaler = StandardScaler().fit(train_rows["daysUntilHoliday"].to_numpy(dtype=np.float64).reshape(-1, 1))
    target_transform = TargetTransform.fit(train_rows["speed"].to_numpy(dtype=np.float64), log_target)
    return {"days_until_holiday": days_scaler, "target": target_transform}


class SegmentSpeedDataset(Dataset):
    """Each sample is one independent SegmentSpeed row: 7 engineered continuous
    features + 3 categorical indices (fed to model.py's embedding tables) -> speed.
    """

    def __init__(self, df: pd.DataFrame, indices: np.ndarray, scalers: dict):
        rows = df.iloc[indices]
        self.cont = torch.from_numpy(
            continuous_features(
                rows["dayNr"].to_numpy(),
                rows["monthOfYear"].to_numpy(),
                rows["minutesSincDaybreak"].to_numpy(),
                rows["daysUntilHoliday"].to_numpy(),
                scalers["days_until_holiday"],
            )
        )
        self.status = torch.from_numpy(rows["statusEnum"].to_numpy(dtype=np.int64, copy=True))
        self.holiday_type = torch.from_numpy(rows["holidayType"].to_numpy(dtype=np.int64, copy=True))
        self.holiday_num = torch.from_numpy(rows["holidayNum"].to_numpy(dtype=np.int64, copy=True))
        speed_kmh = rows["speed"].to_numpy(dtype=np.float32, copy=True)
        self.target_kmh = torch.from_numpy(speed_kmh)
        self.target_scaled = torch.from_numpy(scalers["target"].transform(speed_kmh))

    def __len__(self) -> int:
        return self.cont.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "cont": self.cont[idx],
            "status": self.status[idx],
            "holiday_type": self.holiday_type[idx],
            "holiday_num": self.holiday_num[idx],
            "target_scaled": self.target_scaled[idx],
            "target_kmh": self.target_kmh[idx],
        }


CONTINUOUS_DIM = CONTINUOUS_FEATURE_DIM
