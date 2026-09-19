"""Export raw-trafic-data.json from a running TrafficML backend.

Replaces the ad hoc `curl | jq | sed >> raw-trafic-data.json` approach (see
gen_raw-trafic-data.sh at the repo root): that pattern only ever pulled a single
section and its sed-based array-concatenation trick doesn't produce valid JSON. This
walks every known section from `/api/v1/sections` and writes one valid JSON array.

Stdlib only (urllib) -- runs without the venv/requirements.txt needed for training.
Identical to MachineLearning_MK-1's script of the same name; duplicated here rather
than imported so each MK-* directory stays self-contained and independently runnable.

    python export_raw_trafic_data.py --base-url http://localhost:8080 --out raw-trafic-data.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://localhost:8080")
    p.add_argument("--out", default="raw-trafic-data.json")
    p.add_argument("--page-size-speed", type=int, default=100_000, help="large enough to fetch all speed/records in one page")
    p.add_argument("--page-size-section", type=int, default=100, help="large enough to fetch all sections/records in one page")

    p.add_argument("--limit-sections", type=int, default=None, help="only export the first N sections (for testing this script)")
    args = p.parse_args()

    sections_url = f"{args.base_url}/api/v1/sections?page=0&size={args.page_size_section}"
    print(f"Fetching section list: {sections_url} of size {args.page_size_section}")
    sections = get_json(sections_url)
    section_ids = [s["id"] for s in sections]
    if args.limit_sections:
        section_ids = section_ids[: args.limit_sections]
    print(f"{len(section_ids)} sections to export each with max size of {args.page_size_speed}")

    all_records = []
    failed = []
    for i, section_id in enumerate(section_ids, 1):
        pages_to_fetch = args.page_size_speed
        page_nr = 0
        loaded_pages = 0
        while pages_to_fetch > 0:
            url = f"{args.base_url}/api/v1/sections/{section_id}/ml-speed-data?page={page_nr}&size={pages_to_fetch}"
            #print(f"requesting speed records from url: {url}")
            try:
                records = get_json(url)
                if len(records) == 0:
                    pages_to_fetch = 0
                else:
                    pages_to_fetch = pages_to_fetch - len(records)
                    page_nr = page_nr + 1
                    loaded_pages = loaded_pages + len(records)
            except urllib.error.URLError as exc:
                print(f"  [{i}/{len(section_ids)}] section {section_id}: FAILED ({exc})", file=sys.stderr)
                failed.append(section_id)
                pages_to_fetch = 0
                page_nr = 0
                continue
            all_records.extend(records)


        if i % 25 == 0 or i == len(section_ids):
            print(f"  [{i}/{len(section_ids)}] section {section_id}: {loaded_pages} records (total so far: {len(all_records)})")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_records, f)

    print(f"\nWrote {len(all_records)} records for {len(section_ids) - len(failed)} sections to {args.out}")
    if failed:
        print(f"{len(failed)} sections failed and were skipped: {failed}", file=sys.stderr)


if __name__ == "__main__":
    main()
