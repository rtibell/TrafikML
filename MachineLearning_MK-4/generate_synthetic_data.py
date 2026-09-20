"""Generate a small synthetic raw-trafic-data.json for smoke-testing the pipeline
(dataset -> model -> train -> tune -> evaluate) without needing a real export from the
TrafficML collector. Not a substitute for training on real trafiken.nu data --
holiday/calendar fields here are simplified stand-ins, not the real Swedish holiday
calendar computed by SwedishHolidays.java.

Rows here are independent samples (no per-segment minute-by-minute continuity) since
this model consumes each SegmentSpeed row independently -- see dataset.py.

    python generate_synthetic_data.py --out synthetic-trafic-data.json --rows 20000

Deliberately does NOT default to raw-trafic-data.json -- that filename is reserved for
the real export from the running collector; never overwrite it with synthetic data.
"""

from __future__ import annotations

import argparse
import json
import math
import random

from features import HOLIDAY_IS_FIELDS, HOLIDAY_TYPE_FIELDS, STATUS_FIELDS, STATUS_VALUES


def speed_to_status(speed: float) -> int:
    if speed >= 50:
        return 0
    if speed >= 30:
        return 1
    if speed >= 10:
        return 2
    return 3


def one_hot(fields: list[str], selected_index: int) -> dict:
    return {field: (1 if i == selected_index else 0) for i, field in enumerate(fields)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="synthetic-trafic-data.json")
    p.add_argument("--rows", type=int, default=20000)
    p.add_argument("--num-segments", type=int, default=20)
    p.add_argument("--seed", type=int, default=4711)
    args = p.parse_args()

    rng = random.Random(args.seed)
    records = []
    for _ in range(args.rows):
        day_nr = rng.randrange(7)
        month = rng.randrange(1, 13)
        minutes_since_daybreak = rng.randrange(0, 1440)
        days_until_holiday = rng.randrange(0, 60)
        holiday_type_index = 0 if days_until_holiday > 1 else rng.choice([1, 2])
        holiday_is_index = rng.randrange(len(HOLIDAY_IS_FIELDS))

        is_weekday = day_nr < 5
        rush_am = math.exp(-((minutes_since_daybreak - 450) ** 2) / (2 * 40 ** 2))
        rush_pm = math.exp(-((minutes_since_daybreak - 1020) ** 2) / (2 * 50 ** 2))
        congestion = (rush_am + rush_pm) if is_weekday else 0.15 * (rush_am + rush_pm)
        congestion = min(congestion, 1.0)

        free_flow_speed = 70.0
        speed = free_flow_speed * (1 - congestion * 0.6)
        speed += rng.gauss(0, 3)
        speed = max(2.0, min(speed, free_flow_speed + 5))
        status_index = speed_to_status(speed)

        record = {
            "sectionId": 30000 + rng.randrange(args.num_segments),
            "measureTime": f"2026-{month:02d}-01T00:00:00",
            "status": STATUS_VALUES[status_index],
            "statusEnum": status_index,
            "speed": round(speed, 1),
            "dayNr": day_nr,
            "daysUntilHoliday": days_until_holiday,
            "minutesSincDaybreak": minutes_since_daybreak,
            "monthOfYear": month,
        }
        record.update(one_hot(HOLIDAY_TYPE_FIELDS, holiday_type_index))
        record.update(one_hot(STATUS_FIELDS, status_index))
        record.update(one_hot(HOLIDAY_IS_FIELDS, holiday_is_index))
        records.append(record)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f)
    print(f"Wrote {len(records)} synthetic rows to {args.out}")


if __name__ == "__main__":
    main()
