"""Generate a small synthetic raw-trafic-data.json for smoke-testing the pipeline
(dataset -> model -> train -> tune -> evaluate) without needing a real export from the
TrafficML collector. Not a substitute for training on real trafiken.nu data --
holiday/calendar fields here are simplified stand-ins, not the real Swedish holiday
calendar computed by SwedishHolidays.java.

Unlike MachineLearning_MK-1's generator, rows here are independent samples (no
per-segment minute-by-minute continuity) since this model consumes each SegmentSpeed
row independently -- see dataset.py.

    python generate_synthetic_data.py --out synthetic-trafic-data.json --rows 20000

Deliberately does NOT default to raw-trafic-data.json -- that filename is reserved for
the real export from the running collector; never overwrite it with synthetic data.
"""

from __future__ import annotations

import argparse
import json
import math
import random

STATUS_NAMES = ["freeflow", "heavy", "congested", "impossible"]


def speed_to_status(speed: float) -> int:
    if speed >= 50:
        return 0
    if speed >= 30:
        return 1
    if speed >= 10:
        return 2
    return 3


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
        holiday_type = 0 if days_until_holiday > 1 else rng.choice([1, 2])
        holiday_num = rng.randrange(1, 15)

        is_weekday = day_nr < 5
        rush_am = math.exp(-((minutes_since_daybreak - 450) ** 2) / (2 * 40 ** 2))
        rush_pm = math.exp(-((minutes_since_daybreak - 1020) ** 2) / (2 * 50 ** 2))
        congestion = (rush_am + rush_pm) if is_weekday else 0.15 * (rush_am + rush_pm)
        congestion = min(congestion, 1.0)

        free_flow_speed = 70.0
        speed = free_flow_speed * (1 - congestion * 0.6)
        speed += rng.gauss(0, 3)
        speed = max(2.0, min(speed, free_flow_speed + 5))

        records.append(
            {
                "sectionId": 30000 + rng.randrange(args.num_segments),
                "measureTime": f"2026-{month:02d}-01T00:00:00",
                "status": STATUS_NAMES[speed_to_status(speed)],
                "statusEnum": speed_to_status(speed),
                "speed": round(speed, 1),
                "dayNr": day_nr,
                "daysUntilHoliday": days_until_holiday,
                "holidayType": holiday_type,
                "minutesSincDaybreak": minutes_since_daybreak,
                "monthOfYear": month,
                "holidayNum": holiday_num,
            }
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f)
    print(f"Wrote {len(records)} synthetic rows to {args.out}")


if __name__ == "__main__":
    main()
