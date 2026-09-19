"""Feature engineering shared by dataset building, training, tuning and evaluation.

Field semantics match the `/api/v1/sections/{id}/ml-speed-data` endpoint documented in
REST-opperations.md, with `holidayType` used as the field name given in the task's
SegmentSpeed structure (equivalent to the API's `holidayNr`: 0=regular day, 1=holiday
eve, 2=holiday).
"""

from __future__ import annotations

import math

import numpy as np

STATUS_VALUES = ["freeflow", "heavy", "congested", "impossible"]
STATUS_CARDINALITY = 4          # statusEnum 0..3
HOLIDAY_TYPE_CARDINALITY = 3    # holidayType 0..2
HOLIDAY_NUM_CARDINALITY = 15    # holidayNum 0..14 (0 reserved/unused, real values 1..14)

DAY_NR_PERIOD = 7
MONTH_PERIOD = 12
MINUTES_SINCE_DAYBREAK_PERIOD = 1440


def cyclical_encode(value: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (sin, cos) encoding of a value that wraps around `period`."""
    angle = 2.0 * math.pi * (value.astype(np.float64) / period)
    return np.sin(angle).astype(np.float32), np.cos(angle).astype(np.float32)


def continuous_features(day_nr, month_of_year, minutes_since_daybreak, days_until_holiday, days_scaler) -> np.ndarray:
    """Vectorized continuous covariates for every row.

    Returns an (N, 7) float32 array: sin/cos(dayNr), sin/cos(month), sin/cos(minutes),
    scaled daysUntilHoliday. `statusEnum`, `holidayType` and `holidayNum` are *not*
    included here -- they are small-cardinality categoricals, embedded separately by
    the model (see model.py::TabularCNNRegressor).
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
