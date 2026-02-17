#!/usr/bin/env python3
"""
Pull metadata for a sample of discovered datasets to inspect structure before
locking in eval/regular heuristics and episode-count logic.

Run discovery first (without --fetch-episodes), then run this on the CSV to
download meta/info.json and optionally list repo files for a few repos.
Inspect data/sample_meta/ to see: total_episodes, schema, v2.1 vs v3, naming
patterns for eval vs regular, then tune config and re-run discovery.

Usage:
  python scripts/discover_lerobot_datasets.py --limit 50 -o data/discovery.csv
  python scripts/pull_sample_metadata.py data/discovery.csv -n 15 -o data/sample_meta
  python scripts/pull_sample_metadata.py data/discovery.csv -n 20 --by so101_mentioned
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


def sanitize_repo_id(repo_id: str) -> str:
    return re.sub(r"[^\w\-.]", "_", repo_id)


def pull_meta(repo_id: str, out_dir: Path, list_files: bool, token: bool | str | None = True) -> dict:
    """Download meta/info.json and optionally list repo files. Returns summary dict."""
    safe = sanitize_repo_id(repo_id)
    repo_dir = out_dir / safe
    repo_dir.mkdir(parents=True, exist_ok=True)
    summary = {"repo_id": repo_id, "meta": None, "files_sample": None, "error": None}
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="meta/info.json",
            repo_type="dataset",
            token=token,
        )
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
        summary["meta"] = meta
        # Write a copy for easy inspection
        (repo_dir / "info.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception as e:
        summary["error"] = str(e)
        return summary
    if list_files:
        try:
            files = list_repo_files(repo_id, repo_type="dataset", token=token)
            summary["files_sample"] = files[:80]
            (repo_dir / "repo_files.txt").write_text("\n".join(files), encoding="utf-8")
        except Exception as e:
            summary["error"] = summary.get("error") or "" + f"; list_files: {e}"
    return summary


def main():
    p = argparse.ArgumentParser(description="Pull metadata sample from discovery CSV for inspection")
    p.add_argument("csv_path", type=Path, help="Discovery results CSV (from discover_lerobot_datasets.py)")
    p.add_argument("-n", "--num", type=int, default=15, help="Number of repos to sample (default 15)")
    p.add_argument("-o", "--output-dir", type=Path, default=Path("data/sample_meta"),
                    help="Output directory for meta copies (default data/sample_meta)")
    p.add_argument("--by", type=str, default="downloads",
                    choices=["downloads", "so101_mentioned", "total_task_score", "first"],
                    help="Sample order: first N by this column (default downloads)")
    p.add_argument("--list-files", action="store_true", help="Also list repo files (more API calls)")
    p.add_argument("--delay", type=float, default=0.3, help="Seconds between repo requests (default 0.3)")
    args = p.parse_args()

    if not args.csv_path.exists():
        p.error(f"CSV not found: {args.csv_path}")

    rows = []
    with open(args.csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("CSV is empty.")
        return

    # Sort and take first N
    key = args.by
    if key == "so101_mentioned":
        rows = sorted(rows, key=lambda r: (r.get(key, "").lower() == "true", int(r.get("downloads", 0) or 0)), reverse=True)
    elif key == "total_task_score":
        rows = sorted(rows, key=lambda r: int(r.get(key, 0) or 0), reverse=True)
    elif key == "first":
        pass
    else:
        rows = sorted(rows, key=lambda r: int(r.get(key, 0) or 0), reverse=True)
    sample = rows[: args.num]
    repo_ids = [r["repo_id"] for r in sample]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for i, repo_id in enumerate(repo_ids):
        print(f"[{i+1}/{len(repo_ids)}] {repo_id}")
        s = pull_meta(repo_id, args.output_dir, list_files=args.list_files)
        summaries.append(s)
        if args.delay > 0:
            time.sleep(args.delay)

    # Write a short summary index
    index = []
    for s in summaries:
        n = s["meta"].get("total_episodes") if s.get("meta") else None
        err = s.get("error") or ""
        index.append({"repo_id": s["repo_id"], "total_episodes": n, "error": err})
    (args.output_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_dir}; index in index.json")
    with_episodes = sum(1 for s in index if s.get("total_episodes") is not None)
    print(f"  Has total_episodes: {with_episodes}/{len(index)}")
