#!/usr/bin/env python3
"""
Enrich dataset CSVs with camera configuration from meta/info.json.

For each dataset, downloads meta/info.json and extracts observation.images.*
keys (where dtype == "video") to determine the camera setup. Outputs:
  1. Enriched CSV with new columns: num_cameras, camera_names, camera_config
  2. Summary text file grouping datasets by camera configuration

Resume support:
  --resume  skips repo_ids already in the output CSV

Usage:
  python scripts/enrich_cameras.py -i data/author_expansion.csv -o data/author_expansion_cameras.csv
  python scripts/enrich_cameras.py --resume -i data/author_expansion.csv -o data/author_expansion_cameras.csv
  # Multiple input CSVs:
  python scripts/enrich_cameras.py -i data/author_expansion.csv data/author_expansion_v21.csv -o data/all_cameras.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_existing_csv(path: Path) -> tuple[list[dict], set[str]]:
    """Load existing CSV -> (rows, set of repo_ids)."""
    if not path.exists():
        return [], set()
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("repo_id"):
                rows.append(row)
    return rows, {r["repo_id"] for r in rows}


def get_camera_info(repo_id: str, token: bool | str | None = True, timeout: int = 20) -> dict | None:
    """Download meta/info.json and extract camera names from features.

    Returns dict with:
        cameras: sorted list of camera names (e.g. ["top", "wrist"])
    or None on failure/timeout.
    """
    result = [None]
    error = [None]

    def _download():
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename="meta/info.json",
                repo_type="dataset",
                token=token,
            )
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            features = data.get("features", {})
            cameras = []
            for key, val in features.items():
                if key.startswith("observation.images."):
                    # Only count video features as cameras
                    if isinstance(val, dict) and val.get("dtype") == "video":
                        cam_name = key.replace("observation.images.", "")
                        cameras.append(cam_name)
            cameras.sort()
            result[0] = {"cameras": cameras}
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_download, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        print(f"    TIMEOUT meta for {repo_id} (>{timeout}s)")
        return None
    if error[0]:
        return None
    return result[0]


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def enrich(
    input_rows: list[dict],
    output_path: Path,
    summary_path: Path,
    resume_rows: list[dict] | None = None,
    resume_repo_ids: set[str] | None = None,
    delay: float = 0.25,
) -> list[dict]:
    is_resume = resume_rows is not None
    rows: list[dict] = list(resume_rows) if resume_rows else []
    seen: set[str] = set(resume_repo_ids) if resume_repo_ids else set()

    # Determine output fieldnames: input columns + new camera columns
    input_fields = list(input_rows[0].keys()) if input_rows else []
    new_fields = ["num_cameras", "camera_names", "camera_config"]
    out_fields = input_fields + new_fields

    # Open CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if is_resume:
        csv_file = open(output_path, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=out_fields)
    else:
        csv_file = open(output_path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=out_fields)
        writer.writeheader()
    csv_file.flush()

    total = len(input_rows)
    processed = 0
    skipped = 0

    try:
        for i, row in enumerate(input_rows, 1):
            repo_id = row["repo_id"]
            if repo_id in seen:
                skipped += 1
                continue
            seen.add(repo_id)

            processed += 1
            print(f"  [{processed}/{total - skipped}] {repo_id} ...", end=" ", flush=True)

            # Retry up to 2 times
            cam_info = None
            for attempt in range(1, 3):
                cam_info = get_camera_info(repo_id)
                if cam_info is not None:
                    break
                if attempt < 2:
                    time.sleep(2)
                    print("retry...", end=" ", flush=True)

            if delay > 0:
                time.sleep(delay)

            if cam_info is None:
                print("no meta / failed")
                # Still add the row but with empty camera info
                out_row = {**row, "num_cameras": "", "camera_names": "", "camera_config": ""}
                rows.append(out_row)
                writer.writerow(out_row)
                csv_file.flush()
                continue

            cameras = cam_info["cameras"]
            num_cams = len(cameras)
            cam_names = ",".join(cameras)
            cam_config = f"{num_cams}cam:{cam_names}" if cameras else "0cam:none"

            print(f"{cam_config}")

            out_row = {**row, "num_cameras": num_cams, "camera_names": cam_names, "camera_config": cam_config}
            rows.append(out_row)
            writer.writerow(out_row)
            csv_file.flush()

    except KeyboardInterrupt:
        print(f"\n\nInterrupted! {len(rows)} rows saved so far.")
        print("  Run again with --resume to continue.")
    finally:
        csv_file.close()

    print(f"\nDone. Processed {processed}, skipped {skipped} (already in output).")
    print(f"Total rows: {len(rows)}")

    # Write summary
    write_summary(rows, summary_path)

    return rows


def write_summary(rows: list[dict], summary_path: Path) -> None:
    """Group datasets by camera_config and write summary file."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        config = r.get("camera_config", "unknown")
        if not config:
            config = "unknown"
        groups[config].append(r)

    # Sort groups by dataset count descending
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Camera Configuration Summary\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Total datasets: {len(rows)}\n")
        f.write(f"Unique camera configs: {len(sorted_groups)}\n\n")

        for config, ds_list in sorted_groups:
            total_eps = 0
            for d in ds_list:
                try:
                    total_eps += int(d.get("total_episodes", 0) or 0)
                except (ValueError, TypeError):
                    pass
            f.write(f"=== {config} ({len(ds_list)} datasets, {total_eps:,} episodes) ===\n")
            for d in ds_list:
                eps = d.get("total_episodes", "?")
                f.write(f"  {d['repo_id']}  ({eps} eps)\n")
            f.write("\n")

    print(f"Summary: {summary_path}")


def main():
    p = argparse.ArgumentParser(description="Enrich dataset CSVs with camera configuration")
    p.add_argument("-i", "--input", type=Path, nargs="+", required=True,
                   help="Input CSV file(s) (e.g. data/author_expansion.csv)")
    p.add_argument("-o", "--output", type=Path, default=Path("data/author_expansion_cameras.csv"),
                   help="Output enriched CSV path")
    p.add_argument("--summary", type=Path, default=None,
                   help="Summary output path (default: <output>.summary.txt)")
    p.add_argument("--delay", type=float, default=0.25,
                   help="Seconds between meta/info.json fetches (default 0.25)")
    p.add_argument("--resume", action="store_true",
                   help="Resume: skip repo_ids already in the output CSV")
    args = p.parse_args()

    summary_path = args.summary or args.output.with_suffix(args.output.suffix + ".summary.txt")

    # Load all input CSVs
    input_rows: list[dict] = []
    seen_input: set[str] = set()
    for csv_path in args.input:
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping")
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rid = row.get("repo_id", "")
                if rid and rid not in seen_input:
                    seen_input.add(rid)
                    input_rows.append(row)
        print(f"Loaded {csv_path}: {len(input_rows)} total unique repos so far")

    if not input_rows:
        print("No input rows found.")
        return

    # Resume
    resume_rows = None
    resume_repo_ids = None
    if args.resume and args.output.exists():
        resume_rows, resume_repo_ids = load_existing_csv(args.output)
        print(f"Resume: {len(resume_rows)} existing rows, {len(resume_repo_ids)} repo_ids to skip")

    print(f"\nEnriching {len(input_rows)} datasets with camera info...")
    print(f"Output: {args.output}\n")

    enrich(
        input_rows=input_rows,
        output_path=args.output,
        summary_path=summary_path,
        resume_rows=resume_rows if args.resume else None,
        resume_repo_ids=resume_repo_ids if args.resume else None,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
