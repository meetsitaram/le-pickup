#!/usr/bin/env python3
"""
Discover SO101-compatible datasets by author expansion.

Strategy:
  1. Search HF for datasets matching eval-like patterns (eval_*, evaluation, etc.)
     — NO LeRobot tag required.
  2. Collect unique authors from the eval hits.
  3. Fetch ALL datasets for each author.
  4. For every dataset, download meta/info.json and check:
     - codebase_version == v3.0
     - robot_type in target list
     - total_episodes >= min_episodes
     - (optional) video_files_size_in_mb > threshold
  5. Output a flat CSV with metadata columns (no description blob).

Resume support:
  - --resume  : reload existing CSV + state, skip already-processed repos & authors
  - State file (<output>.state.json) tracks completed authors so a crash mid-run
    picks up from the next author, not the beginning.
  - Incremental CSV writes: every match is flushed immediately.
  - Ctrl-C is caught gracefully; progress is saved.

Usage:
  python scripts/discover_by_author.py -o data/author_expansion.csv
  python scripts/discover_by_author.py --min-episodes 5 --min-video-size-mb 0 -o data/author_expansion.csv
  python scripts/discover_by_author.py --extra-searches "so101,so100,lerobot" -o data/author_expansion.csv
  # Resume after interruption:
  python scripts/discover_by_author.py --resume -o data/author_expansion.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TARGET_ROBOT_TYPES = ("so100_follower", "so101_follower", "so_follower", "bi_so_follower", "bi_so101_follower", "xlerobot")
REQUIRE_CODEBASE_V3 = "v3.0"
MIN_EPISODES_DEFAULT = 10

# Search terms used to find eval-like datasets (seed authors)
EVAL_SEARCH_TERMS = [
    "eval_so101",
    "eval_so100",
    "eval so101",
    "eval so100",
    "evaluation so101",
    "evaluation so100",
    "eval_so_follower",
    "so101 eval",
    "so100 eval",
]

TASK_KEYWORDS = {
    "pick_and_place": ["pick", "place", "grasp", "put", "transfer", "stack", "cube", "object"],
    "pour_liquids": ["pour", "liquid", "cup", "bottle", "water", "coffee"],
    "laundry_folding": ["fold", "laundry", "towel", "cloth", "fabric"],
}

# CSV columns
FIELDNAMES = [
    "repo_id", "author", "downloads", "likes", "last_modified", "created_at",
    "is_eval_like",
    "task_pick_and_place", "task_pour_liquids", "task_laundry_folding",
    "best_task", "best_task_score", "total_task_score",
    # meta/info.json fields (flattened)
    "codebase_version", "robot_type",
    "total_episodes", "total_frames", "total_tasks", "total_videos", "total_chunks",
    "chunks_size", "fps",
    "data_files_size_in_mb", "video_files_size_in_mb",
]

EVAL_LIKE_PATTERNS = [
    "eval", "evaluation", "benchmark", "eval-set", "eval_set",
    "trained", "checkpoint", "metrics", "results",
]


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def load_existing_csv(path: Path) -> tuple[list[dict], set[str]]:
    """Load existing CSV → (rows, set of repo_ids)."""
    if not path.exists():
        return [], set()
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("repo_id"):
                rows.append(row)
    return rows, {r["repo_id"] for r in rows}


def load_state(state_path: Path) -> dict:
    """Load state JSON (completed_authors list, seed_authors list)."""
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize(s: str | None) -> str:
    return (s or "").lower().strip()


def is_eval_like(repo_id: str) -> bool:
    t = normalize(repo_id)
    return any(p in t for p in EVAL_LIKE_PATTERNS)


def score_tasks(text: str) -> dict[str, int]:
    t = normalize(text)
    scores = {}
    for task, keywords in TASK_KEYWORDS.items():
        scores[task] = sum(1 for k in keywords if k in t)
    return scores


def get_meta_info(repo_id: str, token: bool | str | None = True, timeout: int = 20) -> dict | None:
    """Download meta/info.json and return all useful fields. None on failure/timeout."""
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

            def _num(key):
                v = data.get(key)
                if v is None:
                    return None
                if isinstance(v, (int, float)):
                    return v
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return None

            result[0] = {
                "codebase_version": (data.get("codebase_version") or "").strip() or None,
                "robot_type": (data.get("robot_type") or "").strip() or None,
                "total_episodes": int(data["total_episodes"]) if data.get("total_episodes") is not None else None,
                "total_frames": _num("total_frames"),
                "total_tasks": _num("total_tasks"),
                "total_videos": _num("total_videos"),
                "total_chunks": _num("total_chunks"),
                "chunks_size": _num("chunks_size"),
                "fps": _num("fps"),
                "data_files_size_in_mb": _num("data_files_size_in_mb"),
                "video_files_size_in_mb": _num("video_files_size_in_mb"),
            }
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
# Phase 1: Find seed authors via eval-like searches
# ---------------------------------------------------------------------------

def find_seed_authors(
    api: HfApi,
    extra_searches: list[str] | None = None,
    limit: int = 500,
    cached_authors: set[str] | None = None,
    since_days: int | None = None,
) -> set[str]:
    """Search HF for eval-like datasets and return unique authors.
    If cached_authors is provided (resume), merge with fresh results.
    If since_days is set, only consider datasets modified within that window."""
    authors: set[str] = set(cached_authors or set())
    search_terms = EVAL_SEARCH_TERMS + (extra_searches or [])
    seen_repos: set[str] = set()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=since_days)) if since_days else None

    for term in search_terms:
        print(f"  Searching: '{term}' ...", end=" ", flush=True)
        count = 0
        skipped_old = 0
        try:
            for ds in api.list_datasets(search=term, sort="last_modified", limit=limit):
                if ds.id in seen_repos:
                    continue
                seen_repos.add(ds.id)
                # Date filter: skip datasets older than cutoff
                last_modified = getattr(ds, "last_modified", None)
                if cutoff and last_modified and last_modified < cutoff:
                    skipped_old += 1
                    continue
                author = getattr(ds, "author", "") or (ds.id.split("/")[0] if "/" in ds.id else "")
                if author:
                    authors.add(author)
                count += 1
        except Exception as e:
            print(f"error: {e}")
            continue
        suffix = f" (skipped {skipped_old} older)" if skipped_old else ""
        print(f"{count} recent datasets, {len(authors)} unique authors so far{suffix}")

    print(f"\n  Total seed authors: {len(authors)}")
    return authors


# ---------------------------------------------------------------------------
# Phase 2: For each author, list all datasets and check meta/info.json
# ---------------------------------------------------------------------------

def expand_authors(
    api: HfApi,
    authors: set[str],
    target_robot_types: tuple[str, ...],
    codebase_version: str,
    min_episodes: int,
    min_video_size_mb: float | None,
    since_days: int | None,
    output_path: Path,
    state_path: Path,
    delay: float = 0.25,
    resume_rows: list[dict] | None = None,
    resume_repo_ids: set[str] | None = None,
    completed_authors: set[str] | None = None,
    all_seed_authors: set[str] | None = None,
) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=since_days)) if since_days else None

    # Resume: keep existing rows, open CSV in append mode
    is_resume = resume_rows is not None
    rows: list[dict] = list(resume_rows) if resume_rows else []
    seen: set[str] = set(resume_repo_ids) if resume_repo_ids else set()
    done_authors: set[str] = set(completed_authors) if completed_authors else set()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if is_resume:
        csv_file = open(output_path, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        # Don't rewrite header in append mode
    else:
        csv_file = open(output_path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
    csv_file.flush()

    total_checked = 0
    total_matched = len(rows)

    sorted_authors = sorted(authors)
    try:
        for ai, author in enumerate(sorted_authors, 1):
            if author in done_authors:
                print(f"\n[{ai}/{len(sorted_authors)}] Author: {author} — already done, skipping")
                continue

            print(f"\n[{ai}/{len(sorted_authors)}] Author: {author}")

            # Retry listing author's datasets up to 3 times
            datasets = None
            for attempt in range(1, 4):
                try:
                    datasets = list(api.list_datasets(author=author, sort="last_modified", limit=500))
                    break
                except Exception as e:
                    wait = min(2 ** attempt, 15)
                    print(f"  Error listing datasets (attempt {attempt}/3): {e}")
                    if attempt < 3:
                        print(f"  Retrying in {wait}s...")
                        time.sleep(wait)
            if datasets is None:
                print(f"  Skipping author {author} after 3 failures")
                continue

            print(f"  {len(datasets)} datasets")

            for ds in datasets:
                repo_id = ds.id
                if repo_id in seen:
                    continue
                seen.add(repo_id)

                last_modified = getattr(ds, "last_modified", None)
                if cutoff and last_modified and last_modified < cutoff:
                    continue

                total_checked += 1
                print(f"    [{total_checked}] {repo_id} ...", end=" ", flush=True)

                # Retry meta fetch up to 2 times
                meta = None
                for attempt in range(1, 3):
                    meta = get_meta_info(repo_id)
                    if meta is not None:
                        break
                    if attempt < 2:
                        time.sleep(2)
                        print("retry...", end=" ", flush=True)

                if delay > 0:
                    time.sleep(delay)

                if meta is None:
                    print("no meta")
                    continue

                cv = meta.get("codebase_version")
                rt = meta.get("robot_type")
                eps = meta.get("total_episodes")
                vid_mb = meta.get("video_files_size_in_mb")

                # Filter: v3.0 + target robot
                if codebase_version != "any" and cv != codebase_version:
                    print(f"skip ({cv}, {rt})")
                    continue
                if rt not in target_robot_types:
                    print(f"skip ({cv}, {rt})")
                    continue

                # Filter: min episodes
                if min_episodes > 0 and eps is not None and eps < min_episodes:
                    print(f"too few eps ({eps})")
                    continue

                # Filter: min video size
                if min_video_size_mb is not None:
                    if vid_mb is None or vid_mb <= min_video_size_mb:
                        print(f"video too small ({vid_mb}MB)")
                        continue

                total_matched += 1
                print(f"MATCH #{total_matched} ({rt}, {eps} eps, {vid_mb}MB)")

                # Task scoring from repo name
                task_scores = score_tasks(repo_id)
                total_task = sum(task_scores.values())
                best_task = max(task_scores, key=task_scores.get)
                best_score = task_scores.get(best_task, 0)

                created_at = getattr(ds, "created_at", None)
                row = {
                    "repo_id": repo_id,
                    "author": author,
                    "downloads": getattr(ds, "downloads", None) or 0,
                    "likes": getattr(ds, "likes", None) or 0,
                    "last_modified": last_modified.isoformat() if last_modified else "",
                    "created_at": created_at.isoformat() if created_at else "",
                    "is_eval_like": is_eval_like(repo_id),
                    "task_pick_and_place": task_scores["pick_and_place"],
                    "task_pour_liquids": task_scores["pour_liquids"],
                    "task_laundry_folding": task_scores["laundry_folding"],
                    "best_task": best_task,
                    "best_task_score": best_score,
                    "total_task_score": total_task,
                }
                for key in ["codebase_version", "robot_type", "total_episodes", "total_frames",
                            "total_tasks", "total_videos", "total_chunks", "chunks_size", "fps",
                            "data_files_size_in_mb", "video_files_size_in_mb"]:
                    row[key] = meta.get(key, "")

                rows.append(row)
                writer.writerow(row)
                csv_file.flush()

            # Mark author complete in state
            done_authors.add(author)
            save_state(state_path, {
                "completed_authors": sorted(done_authors),
                "seed_authors": sorted(all_seed_authors or authors),
            })

    except KeyboardInterrupt:
        print(f"\n\nInterrupted! {total_matched} matches saved so far.")
        print(f"  Completed authors: {len(done_authors)}/{len(sorted_authors)}")
        print(f"  Run again with --resume to continue.")
    finally:
        csv_file.close()
        # Save state on any exit
        save_state(state_path, {
            "completed_authors": sorted(done_authors),
            "seed_authors": sorted(all_seed_authors or authors),
        })

    print(f"\nDone. Checked {total_checked} datasets, matched {total_matched}.")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Discover SO101 datasets by author expansion from eval repos")
    p.add_argument("-o", "--output", type=Path, default=Path("data/author_expansion.csv"))
    p.add_argument("--min-episodes", type=int, default=MIN_EPISODES_DEFAULT)
    p.add_argument("--min-video-size-mb", type=float, default=None,
                   help="Require video_files_size_in_mb > this")
    p.add_argument("--since-days", type=int, default=90,
                   help="Only datasets with last_modified within N days (default: 90 = ~3 months; 0 to disable)")
    p.add_argument("--codebase-version", type=str, default="v3.0",
                   help="Required codebase_version from meta/info.json (default: v3.0; use v2.1 for older datasets)")
    p.add_argument("--robot-types", type=str,
                   default=",".join(TARGET_ROBOT_TYPES),
                   help="Comma-separated target robot_type values")
    p.add_argument("--extra-searches", type=str, default=None,
                   help="Extra comma-separated search terms for seed author discovery (e.g. 'so101,lerobot,so100')")
    p.add_argument("--delay", type=float, default=0.25,
                   help="Seconds between meta/info.json fetches (default 0.25)")
    p.add_argument("--search-limit", type=int, default=500,
                   help="Max results per search term (default 500)")
    p.add_argument("--resume", action="store_true",
                   help="Resume: load existing CSV + state, skip completed authors & repos")
    args = p.parse_args()

    robot_types = tuple(t.strip() for t in args.robot_types.split(",") if t.strip())
    extra = [s.strip() for s in args.extra_searches.split(",") if s.strip()] if args.extra_searches else None
    state_path = args.output.with_suffix(args.output.suffix + ".state.json")

    api = HfApi()

    # Load resume state
    resume_rows: list[dict] | None = None
    resume_repo_ids: set[str] | None = None
    completed_authors: set[str] = set()
    cached_seed_authors: set[str] = set()

    if args.resume:
        state = load_state(state_path)
        completed_authors = set(state.get("completed_authors", []))
        cached_seed_authors = set(state.get("seed_authors", []))
        if args.output.exists():
            resume_rows, resume_repo_ids = load_existing_csv(args.output)
            print(f"Resume: {len(resume_rows)} existing rows, {len(resume_repo_ids)} repo_ids to skip")
            print(f"  {len(completed_authors)} authors already completed")
        else:
            resume_rows, resume_repo_ids = [], set()

    print("=" * 60)
    print("Phase 1: Finding seed authors from eval-like searches")
    print("=" * 60)
    since = args.since_days if args.since_days else None
    authors = find_seed_authors(
        api, extra_searches=extra, limit=args.search_limit,
        cached_authors=cached_seed_authors if args.resume else None,
        since_days=since,
    )

    if not authors:
        print("No authors found. Try broadening search terms with --extra-searches.")
        return

    # Save seed authors to state immediately
    save_state(state_path, {
        "completed_authors": sorted(completed_authors),
        "seed_authors": sorted(authors),
    })

    print()
    print("=" * 60)
    remaining = len(authors) - len(completed_authors)
    print(f"Phase 2: Expanding {len(authors)} authors ({remaining} remaining) — checking all their datasets")
    print("=" * 60)
    rows = expand_authors(
        api=api,
        authors=authors,
        target_robot_types=robot_types,
        codebase_version=args.codebase_version,
        min_episodes=args.min_episodes,
        min_video_size_mb=args.min_video_size_mb,
        since_days=since,
        output_path=args.output,
        state_path=state_path,
        delay=args.delay,
        resume_rows=resume_rows if args.resume else None,
        resume_repo_ids=resume_repo_ids if args.resume else None,
        completed_authors=completed_authors,
        all_seed_authors=authors,
    )

    if not rows:
        print("No matching datasets found.")
        return

    # Summary
    eval_count = sum(1 for r in rows if r.get("is_eval_like") in (True, "True"))
    regular_count = len(rows) - eval_count
    unique_authors = len({r["author"] for r in rows})
    with_task = sum(1 for r in rows if int(r.get("total_task_score", 0) or 0) > 0)
    print(f"\nResults: {len(rows)} datasets from {unique_authors} authors")
    print(f"  Eval-like: {eval_count}")
    print(f"  Regular (training data): {regular_count}")
    print(f"  With task keyword match: {with_task}")
    print(f"\nSaved to {args.output}")
    print(f"State: {state_path}")


if __name__ == "__main__":
    main()
