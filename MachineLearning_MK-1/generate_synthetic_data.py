"""Generate a small synthetic raw-trafic-data.json for smoke-testing the pipeline
(dataset -> model -> train -> evaluate) without needing a real export from the
TrafficML collector. Not a substitute for training on real trafiken.nu data --
holiday/calendar fields here are simplified stand-ins, not the real Swedish holiday
calendar computed by SwedishHolidays.java.

    python generate_synthetic_data.py --out synthetic-trafic-data.json

Deliberately does NOT default to raw-trafic-data.json -- that filename is reserved for
the real export from the running collector; never overwrite it with synthetic data.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timedelta

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
    p.add_argument("--num-segments", type=int, default=20)
    p.add_argument("--days", type=int, default=10)
    p.add_argument("--seed", type=int, default=4711)
    args = p.parse_args()

    rng = random.Random(args.seed)
    start = datetime(2026, 8, 1, 0, 0)
    n_minutes = args.days * 24 * 60

    records = []
    for section_id in range(30000, 30000 + args.num_segments):
        free_flow_speed = rng.uniform(50, 90)
        rush_dip = rng.uniform(0.3, 0.7)  # fraction of free-flow speed kept during rush hour
        phase_noise = rng.uniform(-10, 10)

        for m in range(n_minutes):
            ts = start + timedelta(minutes=m)
            minute_of_day = ts.hour * 60 + ts.minute
            is_weekday = ts.weekday() < 5

            rush_am = math.exp(-((minute_of_day - (450 + phase_noise)) ** 2) / (2 * 40 ** 2))
            rush_pm = math.exp(-((minute_of_day - (1020 + phase_noise)) ** 2) / (2 * 50 ** 2))
            congestion = (rush_am + rush_pm) if is_weekday else 0.15 * (rush_am + rush_pm)
            congestion = min(congestion, 1.0)

            speed = free_flow_speed * (1 - congestion * (1 - rush_dip))
            speed += rng.gauss(0, 3)
            speed = max(2.0, min(speed, free_flow_speed + 5))

            daybreak = ts.replace(hour=6, minute=0, second=0, microsecond=0)
            if ts < daybreak:
                daybreak -= timedelta(days=1)
            minutes_since_daybreak = int((ts - daybreak).total_seconds() // 60)

            records.append(
                {
                    "sectionId": section_id,
                    "measureTime": ts.strftime("%Y-%m-%dT%H:%M:%S"),
                    "status": STATUS_NAMES[speed_to_status(speed)],
                    "statusEnum": speed_to_status(speed),
                    "speed": round(speed, 1),
                    "dayNr": ts.weekday(),
                    "daysUntilHoliday": (13 - ts.timetuple().tm_yday % 14),
                    "holidayType": 0,
                    "minutesSincDaybreak": minutes_since_daybreak,
                    "monthOfYear": ts.month,
                    "holidayNum": 1 + (ts.timetuple().tm_yday % 14),
                }
            )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f)
    print(f"Wrote {len(records)} synthetic records for {args.num_segments} segments to {args.out}")


if __name__ == "__main__":
    main()
