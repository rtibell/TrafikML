"""Feature engineering shared by dataset building, training, tuning and evaluation.

Field semantics match the `/api/v1/sections/{id}/ml-speed-data` endpoint documented in
REST-opperations.md. As of this version the endpoint sends `holidayType` and
`statusEnum`/`holidayNum` pre-expanded into one-hot 0/1 fields (`holiday_type_*`,
`*_status`, `holiday_is_*`) rather than as small-integer categoricals -- see
Solution.md's "Data contract" for the full field table and why this removes the
embedding tables earlier versions of this model needed.
"""

from __future__ import annotations

import math

import numpy as np

STATUS_VALUES = ["freeflow", "heavy", "congested", "impossible"]

# One-hot field groups, in the exact order given in the task's SegmentSpeed feature
# list. Order only matters for readability/reproducibility of the assembled feature
# vector -- a plain concatenation doesn't care which one-hot group comes first.
HOLIDAY_TYPE_FIELDS = ["holiday_type_regular_day", "holiday_type_eve", "holiday_type_holiday"]
STATUS_FIELDS = ["freeflow_status", "heavy_status", "congested_status", "imposible_status"]
HOLIDAY_IS_FIELDS = [
    "holiday_is_nyar", "holiday_is_jul", "holiday_is_forsta_maj", "holiday_is_nationaldagen",
    "holiday_is_pask", "holiday_is_kristihimmelsfard", "holiday_is_pingst", "holiday_is_midsommar",
    "holiday_is_allahelgona", "holiday_is_trettondag",
]
ONE_HOT_FIELDS = HOLIDAY_TYPE_FIELDS + STATUS_FIELDS + HOLIDAY_IS_FIELDS
ONE_HOT_FEATURE_DIM = len(ONE_HOT_FIELDS)  # 17

DAY_NR_PERIOD = 7
MONTH_PERIOD = 12
MINUTES_SINCE_DAYBREAK_PERIOD = 1440


def cyclical_encode(value: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (sin, cos) encoding of a value that wraps around `period`."""
    angle = 2.0 * math.pi * (value.astype(np.float64) / period)
    return np.sin(angle).astype(np.float32), np.cos(angle).astype(np.float32)


def one_hot_features(rows) -> np.ndarray:
    """Pass the 17 upstream one-hot fields straight through as an (N, 17) float32
    array -- already 0/1, so no further encoding is needed on this side.
    """
    return rows[ONE_HOT_FIELDS].to_numpy(dtype=np.float32)


def continuous_features(day_nr, month_of_year, minutes_since_daybreak, days_until_holiday, days_scaler) -> np.ndarray:
    """Vectorized cyclical/scaled covariates for every row.

    Returns an (N, 7) float32 array: sin/cos(dayNr), sin/cos(month), sin/cos(minutes),
    scaled daysUntilHoliday. See `one_hot_features` above for the other 17 columns of
    the model's full input vector (`dataset.py::FEATURE_DIM`).
    """
    day_sin, day_cos = cyclical_encode(np.asarray(day_nr), DAY_NR_PERIOD)
    month_sin, month_cos = cyclical_encode(np.asarray(month_of_year), MONTH_PERIOD)
    min_sin, min_cos = cyclical_encode(np.asarray(minutes_since_daybreak), MINUTES_SINCE_DAYBREAK_PERIOD)
    days_scaled = days_scaler.transform(np.asarray(days_until_holiday, dtype=np.float64).reshape(-1, 1)).astype(
        np.float32
    ).reshape(-1)
    return np.stack([day_sin, day_cos, month_sin, month_cos, min_sin, min_cos, days_scaled], axis=1)


# Number of columns produced by continuous_features()
CONTINUOUS_FEATURE_DIM = 7

# Full model input width: one_hot_features() (17) ++ continuous_features() (7)
FEATURE_DIM = ONE_HOT_FEATURE_DIM + CONTINUOUS_FEATURE_DIM  # 24
