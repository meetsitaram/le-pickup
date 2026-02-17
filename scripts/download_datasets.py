#!/usr/bin/env python3
"""
Download LeRobot datasets to a local directory.

Reads a CSV file and downloads all datasets matching specified camera configs
(or all datasets if no filter is given). Uses huggingface_hub for downloads
with resume support and incremental progress tracking.

Usage:
  # Download specific 3-camera configs:
  python scripts/download_datasets.py -i data/author_expansion_cameras.csv --camera-configs "3cam:side,top,wrist" "3cam:front,gripper,top" "3cam:global,top,wrist"

  # Download all datasets in a CSV:
  python scripts/download_datasets.py -i data/author_expansion_cameras.csv

  # Resume after interruption:
  python scripts/download_datasets.py --resume -i data/author_expansion_cameras.csv --camera-configs "3cam:side,top,wrist"
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from huggingface_hub import snapshot_download


# 3-camera configs with top-like + gripper/wrist-like + fixed view
DEFAULT_3CAM_CONFIGS = [
    "3cam:front,gripper,top",
    "3cam:front,top,wrist",
    "3cam:front,side,top",
    "3cam:global,top,wrist",
    "3cam:bottom,gripper,top",
    "3cam:side,top,wrist",
    "3cam:belly,top,wrist",
]


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def download_datasets(
    datasets: list[dict],
    output_dir: Path,
    state_path: Path,
    completed: set[str] | None = None,
    delay: float = 1.0,
) -> None:
    done: set[str] = set(completed) if completed else set()
    total = len(datasets)
    downloaded = 0
    skipped = 0

    try:
        for i, ds in enumerate(datasets, 1):
            repo_id = ds["repo_id"]
            if repo_id in done:
                skipped += 1
                print(f"  [{i}/{total}] {repo_id} — already downloaded, skipping")
                continue

            cam_config = ds.get("camera_config", "unknown")
            eps = ds.get("total_episodes", "?")
            print(f"  [{i}/{total}] Downloading {repo_id} ({cam_config}, {eps} eps)...", flush=True)

            try:
                local_path = snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    local_dir=str(output_dir / repo_id.replace("/", "__")),
                    token=True,
                )
                downloaded += 1
                done.add(repo_id)
                save_state(state_path, {"completed": sorted(done)})
                print(f"    -> {local_path}")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"    ERROR: {type(e).__name__}: {e}")
                print(f"    Skipping {repo_id}, will retry on next --resume run")

            if delay > 0:
                time.sleep(delay)

    except KeyboardInterrupt:
        print(f"\n\nInterrupted! Downloaded {downloaded} datasets so far.")
        print(f"  Run again with --resume to continue.")
    finally:
        save_state(state_path, {"completed": sorted(done)})

    print(f"\nDone. Downloaded {downloaded}, skipped {skipped} (already done).")
    print(f"Total completed: {len(done)}/{total}")


def main():
    p = argparse.ArgumentParser(description="Download LeRobot datasets to local directory")
    p.add_argument("-i", "--input", type=Path, required=True,
                   help="Input CSV with camera_config column (e.g. data/author_expansion_cameras.csv)")
    p.add_argument("-o", "--output-dir", type=Path, default=Path("datasets"),
                   help="Local directory to download datasets into (default: datasets/)")
    p.add_argument("--camera-configs", nargs="*", default=None,
                   help="Filter by camera_config values (e.g. '3cam:side,top,wrist'). "
                        "If not specified and --default-3cam is set, uses the default 3-cam configs. "
                        "If neither is set, downloads ALL datasets in the CSV.")
    p.add_argument("--default-3cam", action="store_true",
                   help="Use the default set of 3-camera configs (front/gripper/top variants)")
    p.add_argument("--delay", type=float, default=1.0,
                   help="Seconds between downloads (default 1.0)")
    p.add_argument("--resume", action="store_true",
                   help="Resume: skip datasets already downloaded")
    p.add_argument("--dry-run", action="store_true",
                   help="List datasets that would be downloaded without actually downloading")
    args = p.parse_args()

    # Determine camera config filter
    configs = None
    if args.camera_configs:
        configs = set(args.camera_configs)
    elif args.default_3cam:
        configs = set(DEFAULT_3CAM_CONFIGS)

    # Load input CSV
    if not args.input.exists():
        print(f"Error: {args.input} not found")
        return

    datasets: list[dict] = []
    seen: set[str] = set()
    with open(args.input, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            repo_id = row.get("repo_id", "")
            if not repo_id or repo_id in seen:
                continue
            seen.add(repo_id)
            if configs and row.get("camera_config", "") not in configs:
                continue
            datasets.append(row)

    if not datasets:
        print("No datasets match the specified filters.")
        return

    # Summary
    total_eps = 0
    for d in datasets:
        try:
            total_eps += int(d.get("total_episodes", 0) or 0)
        except (ValueError, TypeError):
            pass

    config_counts: dict[str, int] = {}
    for d in datasets:
        cc = d.get("camera_config", "unknown")
        config_counts[cc] = config_counts.get(cc, 0) + 1

    print(f"Datasets to download: {len(datasets)} ({total_eps:,} total episodes)")
    print(f"Output directory: {args.output_dir}")
    print(f"\nCamera configs:")
    for cc, cnt in sorted(config_counts.items(), key=lambda x: -x[1]):
        print(f"  {cc}: {cnt} datasets")

    if args.dry_run:
        print(f"\n--- Dry run: listing {len(datasets)} datasets ---")
        for d in datasets:
            print(f"  {d['repo_id']}  ({d.get('camera_config', '?')}, {d.get('total_episodes', '?')} eps)")
        return

    # Resume state
    state_path = args.output_dir / ".download_state.json"
    completed: set[str] = set()
    if args.resume:
        state = load_state(state_path)
        completed = set(state.get("completed", []))
        remaining = len(datasets) - len(completed & seen)
        print(f"\nResume: {len(completed)} already downloaded, {remaining} remaining")

    print()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    download_datasets(
        datasets=datasets,
        output_dir=args.output_dir,
        state_path=state_path,
        completed=completed if args.resume else None,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
