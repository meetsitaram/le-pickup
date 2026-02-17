"""One-off script to clean the description column in an existing CSV."""
import csv
import re
import sys
from pathlib import Path

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/so101_v3_recent_3months.csv")
rows = []
with open(p, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        d = row.get("description", "")
        clean = re.split(r"meta/info\.json|Dataset Structure|\{", d, maxsplit=1)[0]
        clean = " ".join(clean.split()).strip()
        clean = clean.replace('"', "").replace(",", " ")
        row["description"] = clean[:200]
        rows.append(row)

with open(p, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f"Cleaned {len(rows)} rows in {p}")
