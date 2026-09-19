"""Loading raw-trafic-data.json, building sliding-window samples per road segment,
and the 90/10 train/test split (shuffled, seed 4711 per the task spec).
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

from features import CALENDAR_FEATURE_DIM, calendar_features

RANDOM_SEED = 4711


def load_dataframe(json_path: str) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame.from_records(records)
    required = {
        "sectionId", "measureTime", "statusEnum", "speed", "dayNr",
        "daysUntilHoliday", "holidayType", "minutesSincDaybreak", "monthOfYear", "holidayNum",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"raw-trafic-data.json is missing expected fields: {sorted(missing)}")

    df["measureTime"] = pd.to_datetime(df["measureTime"])
    df = df.dropna(subset=["speed", "statusEnum"])
    df = df.sort_values(["sectionId", "measureTime"]).reset_index(drop=True)
    return df


@dataclass
class Window:
    section_idx: int
    start: int  # row index into the per-segment sorted group (encoder start)


class SegmentIndex:
    """Maps sparse sectionId values to contiguous embedding indices."""

    def __init__(self, section_ids: list[int]):
        self.id_to_idx = {sid: i for i, sid in enumerate(sorted(set(section_ids)))}
        self.idx_to_id = {i: sid for sid, i in self.id_to_idx.items()}

    def __len__(self) -> int:
        return len(self.id_to_idx)


def _windows_for_group(n_rows: int, timestamps: np.ndarray, t_in: int, t_out: int, stride: int) -> list[int]:
    """Return valid encoder-start offsets within one segment's rows, requiring the
    full t_in + t_out span to be strictly consecutive one-minute measurements (no gaps).
    """
    span = t_in + t_out
    if n_rows < span:
        return []
    minute_deltas = np.diff(timestamps).astype("timedelta64[s]").astype(np.int64)
    is_break = minute_deltas != 60
    # run_id increases every time there is a gap; rows sharing a run_id are contiguous.
    run_id = np.concatenate(([0], np.cumsum(is_break)))
    starts: list[int] = []
    run_boundaries = np.flatnonzero(np.diff(run_id, prepend=run_id[0] - 1))
    run_boundaries = np.append(run_boundaries, n_rows)
    run_start = 0
    for run_end in run_boundaries[1:]:
        run_len = run_end - run_start
        if run_len >= span:
            last_valid_start = run_end - span
            starts.extend(range(run_start, last_valid_start + 1, stride))
        run_start = run_end
    return starts


def build_windows(df: pd.DataFrame, t_in: int, t_out: int, stride: int) -> tuple[list[Window], SegmentIndex, dict]:
    seg_index = SegmentIndex(df["sectionId"].unique().tolist())
    windows: list[Window] = []
    group_cache: dict[int, pd.DataFrame] = {}
    for section_id, group in df.groupby("sectionId", sort=False):
        group = group.reset_index(drop=True)
        section_idx = seg_index.id_to_idx[section_id]
        group_cache[section_idx] = group
        starts = _windows_for_group(len(group), group["measureTime"].to_numpy(), t_in, t_out, stride)
        windows.extend(Window(section_idx=section_idx, start=s) for s in starts)
    return windows, seg_index, group_cache


def split_train_test(windows: list[Window], test_size: float = 0.1, seed: int = RANDOM_SEED):
    return train_test_split(windows, test_size=test_size, random_state=seed, shuffle=True)


def fit_scalers(windows: list[Window], group_cache: dict, t_in: int, t_out: int) -> dict[str, StandardScaler]:
    """Fit scalers on the *training* windows only, to avoid test-set leakage."""
    speeds = []
    days_until_holiday = []
    for w in windows:
        rows = group_cache[w.section_idx].iloc[w.start : w.start + t_in + t_out]
        speeds.append(rows["speed"].to_numpy(dtype=np.float64))
        days_until_holiday.append(rows["daysUntilHoliday"].to_numpy(dtype=np.float64))
    speed_scaler = StandardScaler().fit(np.concatenate(speeds).reshape(-1, 1))
    days_scaler = StandardScaler().fit(np.concatenate(days_until_holiday).reshape(-1, 1))
    return {"speed": speed_scaler, "days_until_holiday": days_scaler}


class FixedScaler:
    """Minimal drop-in for sklearn's StandardScaler.transform(), reconstructed from
    the mean/scale saved in a training checkpoint so evaluation doesn't need to refit.
    """

    def __init__(self, mean: list[float], scale: list[float]):
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.scale_ = np.asarray(scale, dtype=np.float64)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.scale_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale_ + self.mean_


class TrafficWindowDataset(Dataset):
    """Each sample is one (t_in minutes history -> t_out minutes forecast) window for
    a single road segment, shaped for the encoder-decoder model in model.py.
    """

    def __init__(self, windows: list[Window], group_cache: dict, t_in: int, t_out: int, scalers: dict):
        self.windows = windows
        self.group_cache = group_cache
        self.t_in = t_in
        self.t_out = t_out
        self.speed_scaler = scalers["speed"]
        self.days_scaler = scalers["days_until_holiday"]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        rows = self.group_cache[w.section_idx].iloc[w.start : w.start + self.t_in + self.t_out]

        speed = rows["speed"].to_numpy(dtype=np.float64, copy=True)
        speed_scaled = np.ascontiguousarray(
            self.speed_scaler.transform(speed.reshape(-1, 1)).astype(np.float32).reshape(-1)
        )
        status = rows["statusEnum"].to_numpy(dtype=np.int64, copy=True)

        cal = calendar_features(
            rows["dayNr"].to_numpy(),
            rows["monthOfYear"].to_numpy(),
            rows["minutesSincDaybreak"].to_numpy(),
            rows["daysUntilHoliday"].to_numpy(),
            self.days_scaler,
        )
        holiday_type = rows["holidayType"].to_numpy(dtype=np.int64, copy=True)
        holiday_num = rows["holidayNum"].to_numpy(dtype=np.int64, copy=True)

        t_in, t_out = self.t_in, self.t_out
        sample = {
            "segment_idx": torch.tensor(w.section_idx, dtype=torch.long),
            "enc_speed": torch.from_numpy(speed_scaled[:t_in]),
            "enc_status": torch.from_numpy(status[:t_in]),
            "enc_calendar": torch.from_numpy(cal[:t_in]),
            "enc_holiday_type": torch.from_numpy(holiday_type[:t_in]),
            "enc_holiday_num": torch.from_numpy(holiday_num[:t_in]),
            "dec_calendar": torch.from_numpy(cal[t_in:]),
            "dec_holiday_type": torch.from_numpy(holiday_type[t_in:]),
            "dec_holiday_num": torch.from_numpy(holiday_num[t_in:]),
            "target_speed_scaled": torch.from_numpy(speed_scaled[t_in:]),
            "target_speed_kmh": torch.from_numpy(speed[t_in:].astype(np.float32)),
            "target_status": torch.from_numpy(status[t_in:]),
            "prev_speed_scaled": torch.tensor(speed_scaled[t_in - 1], dtype=torch.float32),
            "prev_status": torch.tensor(status[t_in - 1], dtype=torch.long),
        }
        return sample


CALENDAR_DIM = CALENDAR_FEATURE_DIM
