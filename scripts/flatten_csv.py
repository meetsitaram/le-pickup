"""Extract JSON fields from description column, flatten as new columns, remove description."""
import csv
import json
import re
import sys
from pathlib import Path

INPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/so101_v3_recent_3months copy.csv")
OUTPUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/so101_v3_recent_3months_flat.csv")

# Fields we want to extract from the JSON in the description
META_FIELDS = [
    "total_episodes", "total_frames", "total_tasks", "total_videos",
    "total_chunks", "chunks_size", "data_files_size_in_mb",
    "video_files_size_in_mb", "fps", "splits",
]


def extract_json_from_description(desc: str) -> dict:
    """Try to extract key-value pairs from the JSON blob embedded in the description."""
    result = {}
    if not desc:
        return result
    # Try to find a JSON-like block starting with {
    m = re.search(r'\{.*', desc, re.DOTALL)
    if not m:
        return result
    raw = m.group(0)
    # The JSON is usually truncated; try to parse what we can via regex key-value extraction
    for field in META_FIELDS:
        # Match "field": value (number, string, or object)
        pattern = rf'"{field}"\s*:\s*("([^"]*)"|([\d.]+)|(\{{[^}}]*\}}))'
        fm = re.search(pattern, raw)
        if fm:
            if fm.group(2) is not None:
                result[field] = fm.group(2)
            elif fm.group(3) is not None:
                val = fm.group(3)
                result[field] = int(val) if '.' not in val else float(val)
            elif fm.group(4) is not None:
                result[field] = fm.group(4).replace('"', '')
            else:
                result[field] = fm.group(1)
    return result


rows = []
with open(INPUT, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rows.append(row)

if not rows:
    print("No rows found.")
    sys.exit(1)

# Build output rows
out_rows = []
for row in rows:
    desc = row.pop("description", "")
    meta = extract_json_from_description(desc)
    # Add meta fields (use extracted value, or empty if not found)
    for field in META_FIELDS:
        key = f"meta_{field}"
        row[key] = meta.get(field, "")
    out_rows.append(row)

# Build fieldnames: original minus description, plus meta_ fields at the end
original_fields = [k for k in rows[0].keys() if k != "description"]
# rows[0] already had description popped, so just use its keys
out_fields = list(out_rows[0].keys())

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=out_fields)
    w.writeheader()
    w.writerows(out_rows)

print(f"Wrote {len(out_rows)} rows to {OUTPUT}")
print(f"Columns: {', '.join(out_fields)}")
