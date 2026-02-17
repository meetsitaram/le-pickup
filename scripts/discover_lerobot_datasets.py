#!/usr/bin/env python3
"""
Discover LeRobot datasets on Hugging Face relevant to SO101 and target tasks.

Target tasks: pick_and_place, pour_liquids, laundry_folding.

Recommended two-phase flow:
  1) Run discovery without --fetch-episodes to get a candidate list (fast).
  2) Run scripts/pull_sample_metadata.py on the CSV to pull meta for ~15–20 repos.
  3) Inspect data/sample_meta/ (info.json, index.json) to see real structure:
     total_episodes, schema, v2.1 vs v3, eval vs regular naming.
  4) Tune config (eval patterns, min_episodes) and re-run discovery with
     --fetch-episodes and stricter filters if desired.

- Eval vs regular: Repos whose name/description suggest eval/benchmark/trained
  models are flagged (is_eval_like). author_has_eval marks authors who have
  any eval-like repo (prioritize their regular data).
- Metadata filter: --fetch-episodes reads meta/info.json. By default we keep only
  codebase_version "v3.0", robot_type in so100_follower, so101_follower, and
  (optional) video_files_size_in_mb > --min-video-size-mb. --since-days keeps only
  last_modified within N days. Use --no-robot-filter to keep all with meta.

Rate limits: See docs/RATE_LIMITS.md. Use HF_TOKEN.

Usage:
  python scripts/discover_lerobot_datasets.py --limit 50 -o data/discovery.csv
  python scripts/pull_sample_metadata.py data/discovery.csv -n 15
  python scripts/discover_lerobot_datasets.py --fetch-episodes --min-episodes 10
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

# Task keywords for relevance scoring (from config/task_taxonomy.yaml)
TASK_KEYWORDS = {
    "pick_and_place": ["pick", "place", "grasp", "put", "transfer", "stack", "cube", "object"],
    "pour_liquids": ["pour", "liquid", "cup", "bottle", "water", "coffee"],
    "laundry_folding": ["fold", "laundry", "towel", "cloth", "fabric"],
}
ROBOT_SO101_NAMES = ["so101", "so-101", "SO101", "SO-101", "so_101"]

# Repo name/description patterns that suggest eval/benchmark/trained model (not raw recordings)
EVAL_LIKE_PATTERNS = [
    "eval", "evaluation", "benchmark", "eval-set", "eval_set",
    "trained", "checkpoint", "metrics", "results",
]
MIN_EPISODES_DEFAULT = 10

# Target robot types from meta/info.json (SO100/SO101 family; name may not be in dataset title)
# Includes "so_follower" and "bi_so_follower" as some uploaders use those variants
TARGET_ROBOT_TYPES = ("so100_follower", "so101_follower", "so_follower", "bi_so_follower", "bi_so101_follower", "xlerobot")
REQUIRE_CODEBASE_V3 = "v3.0"


def normalize_text(s: str | None) -> str:
    if s is None:
        return ""
    return (s or "").lower().strip()


def score_task_relevance(text: str) -> dict[str, int]:
    """Return count of keyword hits per task."""
    t = normalize_text(text)
    scores = {}
    for task, keywords in TASK_KEYWORDS.items():
        scores[task] = sum(1 for k in keywords if k in t)
    return scores


def is_so101_mentioned(text: str) -> bool:
    t = normalize_text(text)
    return any(name.lower() in t for name in ROBOT_SO101_NAMES)


def is_eval_like(repo_id: str, description: str) -> bool:
    """True if repo looks like an eval/benchmark/trained model dataset, not raw recordings."""
    t = normalize_text(f"{repo_id} {description}")
    return any(p in t for p in EVAL_LIKE_PATTERNS)


def load_existing_csv(path: Path) -> tuple[list[dict], set[str]]:
    """Load existing CSV and return (list of row dicts, set of repo_ids). Returns ([], set()) if file missing or empty."""
    if not path.exists():
        return [], set()
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("repo_id"):
                rows.append(row)
    return rows, {r["repo_id"] for r in rows}


def get_meta_info(repo_id: str, token: bool | str | None = True, timeout: int = 15) -> dict | None:
    """Read meta/info.json; return total_episodes, codebase_version, robot_type, video_files_size_in_mb, etc.
    Returns None if the file is missing, repo is private, or download times out."""
    import signal
    import threading

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
            v = data.get("video_files_size_in_mb")
            d = data.get("data_files_size_in_mb")
            result[0] = {
                "total_episodes": int(data["total_episodes"]) if data.get("total_episodes") is not None else None,
                "codebase_version": (data.get("codebase_version") or "").strip() or None,
                "robot_type": (data.get("robot_type") or "").strip() or None,
                "video_files_size_in_mb": float(v) if v is not None and isinstance(v, (int, float)) else None,
                "data_files_size_in_mb": float(d) if d is not None and isinstance(d, (int, float)) else None,
            }
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_download, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        print(f"  TIMEOUT fetching meta for {repo_id} (>{timeout}s), skipping")
        return None
    if error[0]:
        return None
    return result[0]


def discover(
    limit: int = 500,
    search: str | None = None,
    filter_lerobot: bool = True,
    sort: str = "downloads",
    fetch_episodes: bool = False,
    min_episodes: int = MIN_EPISODES_DEFAULT,
    episode_fetch_delay: float = 0.25,
    only_target_robot: bool = True,
    target_robot_types: tuple[str, ...] = TARGET_ROBOT_TYPES,
    since_days: int | None = None,
    min_video_size_mb: float | None = None,
    output_path: Path | None = None,
    existing_rows: list[dict] | None = None,
    resume_skip_repo_ids: set[str] | None = None,
    window_days: int | None = None,
    state_path: Path | None = None,
    one_week_only: bool = False,
) -> list[dict]:
    api = HfApi()
    # List datasets with LeRobot tag; optionally narrow by search
    # See https://huggingface.co/datasets?other=LeRobot
    # sort: "downloads" | "likes" (hearts) – Hub API always returns descending
    filters = ["LeRobot"] if filter_lerobot else None
    it = api.list_datasets(
        filter=filters,
        search=search,
        sort=sort,
        limit=limit,
    )
    cutoff_date = (datetime.now(timezone.utc) - timedelta(days=since_days)) if since_days else None
    # Column order for incremental CSV (author_has_eval last, filled at end)
    fieldnames = [
        "repo_id", "author", "downloads", "likes", "last_modified", "created_at", "description",
        "so101_mentioned", "is_eval_like", "task_pick_and_place", "task_pour_liquids", "task_laundry_folding",
        "best_task", "best_task_score", "total_task_score", "num_episodes", "codebase_version", "robot_type",
        "video_files_size_in_mb", "data_files_size_in_mb", "author_has_eval",
    ]
    resume_mode = existing_rows is not None and resume_skip_repo_ids is not None
    rows = list(existing_rows) if existing_rows else []

    # Week-by-week state: we've processed last_modified > completed_through; next window is (completed_through - window_days, completed_through]
    window_start = None
    window_end = None
    if state_path and state_path.exists() and window_days and since_days:
        try:
            s = json.loads(state_path.read_text(encoding="utf-8"))
            end_str = s.get("completed_through")
            if end_str:
                window_end = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                window_start = window_end - timedelta(days=window_days)
        except Exception:
            pass
    if window_end is None and window_days and since_days:
        window_end = datetime.now(timezone.utc)
        window_start = window_end - timedelta(days=window_days)
    if cutoff_date and window_start is not None and window_start < cutoff_date:
        window_start = cutoff_date

    csv_file = None
    csv_writer = None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if resume_mode else "w"
        csv_file = open(output_path, mode, newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not resume_mode:
            csv_writer.writeheader()
        csv_file.flush()
    checked = 0
    matched = 0
    max_retries = 5
    retry_count = 0
    seen_repo_ids = set(resume_skip_repo_ids) if resume_skip_repo_ids else set()
    try:
      while True:
        try:
          for info in it:
            repo_id = info.id
            retry_count = 0  # reset on successful iteration
            if repo_id in seen_repo_ids:
                continue
            seen_repo_ids.add(repo_id)
            if resume_mode and repo_id in resume_skip_repo_ids:
                continue
            last_modified = getattr(info, "last_modified", None)
            if cutoff_date and last_modified and last_modified < cutoff_date:
                continue
            # Week window: only accept last_modified in (window_start, window_end]; advance window when we see older data
            # State "completed_through" = oldest date we've fully processed; next window is (completed_through - window_days, completed_through]
            if window_start is not None and window_end is not None and last_modified:
                while last_modified <= window_start:
                    if state_path:
                        state_path.parent.mkdir(parents=True, exist_ok=True)
                        state_path.write_text(
                            json.dumps({"completed_through": window_start.isoformat()}, indent=2),
                            encoding="utf-8",
                        )
                    if one_week_only:
                        break
                    window_end = window_start
                    window_start = window_end - timedelta(days=window_days)
                    if cutoff_date and window_start < cutoff_date:
                        break
                if one_week_only and last_modified <= window_start:
                    break
                if last_modified <= window_start or last_modified > window_end:
                    continue
            author = getattr(info, "author", "") or (repo_id.split("/")[0] if "/" in repo_id else "")
            description = getattr(info, "description", None) or getattr(info, "cardData", None)
            if isinstance(description, dict):
                description = description.get("description") or description.get("task") or json.dumps(description)
            desc_str = (description or "") if isinstance(description, str) else str(description or "")
            # Single-line summary for CSV (no newlines); full text kept for readability
            desc_one_line = " ".join(desc_str.split())[:500].strip() if desc_str else ""
            combined_text = f"{repo_id} {author} {desc_str}"

            task_scores = score_task_relevance(combined_text)
            so101 = is_so101_mentioned(combined_text)
            total_task_score = sum(task_scores.values())
            best_task = max(task_scores, key=task_scores.get) if task_scores else ""
            best_score = task_scores.get(best_task, 0) if best_task else 0
            eval_like = is_eval_like(repo_id, desc_str)

            num_episodes: int | None = None
            codebase_version: str | None = None
            robot_type: str | None = None
            video_files_size_in_mb: float | None = None
            data_files_size_in_mb: float | None = None
            if fetch_episodes:
                checked += 1
                print(f"  [{checked}] Checking {repo_id} ...", end=" ", flush=True)
                meta = get_meta_info(repo_id)
                if episode_fetch_delay > 0:
                    time.sleep(episode_fetch_delay)
                if meta is None:
                    print("no meta / skipped")
                    continue
                num_episodes = meta["total_episodes"]
                codebase_version = meta["codebase_version"]
                robot_type = meta["robot_type"]
                video_files_size_in_mb = meta.get("video_files_size_in_mb")
                data_files_size_in_mb = meta.get("data_files_size_in_mb")
                if only_target_robot:
                    if codebase_version != REQUIRE_CODEBASE_V3 or robot_type not in target_robot_types:
                        print(f"skip ({codebase_version}, {robot_type})")
                        continue
                if min_episodes > 0 and num_episodes is not None and num_episodes < min_episodes:
                    print(f"too few episodes ({num_episodes})")
                    continue
                if min_video_size_mb is not None:
                    if video_files_size_in_mb is None or video_files_size_in_mb <= min_video_size_mb:
                        print(f"video too small ({video_files_size_in_mb}MB)")
                        continue
                matched += 1
                print(f"MATCH #{matched} ({robot_type}, {num_episodes} eps, {video_files_size_in_mb}MB video)")

            created_at = getattr(info, "created_at", None)
            row = {
                "repo_id": repo_id,
                "author": author,
                "downloads": getattr(info, "downloads", None) or 0,
                "likes": getattr(info, "likes", None) or 0,
                "last_modified": last_modified.isoformat() if last_modified else "",
                "created_at": created_at.isoformat() if created_at else "",
                "description": desc_one_line,
                "so101_mentioned": so101,
                "is_eval_like": eval_like,
                "task_pick_and_place": task_scores["pick_and_place"],
                "task_pour_liquids": task_scores["pour_liquids"],
                "task_laundry_folding": task_scores["laundry_folding"],
                "best_task": best_task,
                "best_task_score": best_score,
                "total_task_score": total_task_score,
                "num_episodes": num_episodes,
                "codebase_version": codebase_version,
                "robot_type": robot_type,
                "video_files_size_in_mb": video_files_size_in_mb,
                "data_files_size_in_mb": data_files_size_in_mb,
            }
            rows.append(row)
            if csv_writer is not None and csv_file is not None:
                row_write = {**row, "author_has_eval": False}
                csv_writer.writerow(row_write)
                csv_file.flush()
          # for-loop ended normally: done
          break
        except KeyboardInterrupt:
          print(f"\nInterrupted by user. {matched} matches saved so far.")
          break
        except Exception as e:
          retry_count += 1
          if retry_count > max_retries:
              print(f"\nMax retries ({max_retries}) exceeded. {matched} matches saved so far. Last error: {e}")
              break
          wait = min(2 ** retry_count, 30)
          print(f"\nNetwork error: {type(e).__name__}: {e}")
          print(f"  Retrying in {wait}s (attempt {retry_count}/{max_retries})...")
          time.sleep(wait)
          # Re-create the iterator for the next attempt
          it = api.list_datasets(filter=filters, search=search, sort=sort, limit=limit)

    finally:
        if csv_file is not None:
            csv_file.close()
            csv_file = None
            csv_writer = None

    # Authors who have at least one eval-like repo: their regular datasets may have good recordings
    authors_with_eval = {r["author"] for r in rows if r["is_eval_like"]}
    for r in rows:
        r["author_has_eval"] = r["author"] in authors_with_eval

    if output_path and rows:
        # Rewrite file with correct author_has_eval (incremental write had False for all)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    return rows


def main():
    p = argparse.ArgumentParser(description="Discover LeRobot datasets for SO101 / pick-pour-fold")
    p.add_argument("--output", "-o", type=Path, default=Path("data/discovery_results.csv"),
                    help="Output CSV path")
    p.add_argument("--json", type=Path, default=None, help="Also write full JSON")
    p.add_argument("--search", type=str, default=None,
                    help="Hugging Face search query (e.g. 'so101', 'pour')")
    p.add_argument("--limit", type=int, default=500, help="Max datasets to fetch")
    p.add_argument("--sort", type=str, default="downloads",
                    choices=["downloads", "likes", "last_modified", "created_at"],
                    help="Hub search order: downloads (default), likes, last_modified (newest updated), created_at (newest first); API returns descending")
    p.add_argument("--no-lerobot-filter", action="store_true", help="Do not filter by LeRobot tag")
    p.add_argument("--fetch-episodes", action="store_true",
                    help="Fetch meta/info.json; keep only datasets with metadata, v3.0, target robot (so100/so101_follower), and >= min-episodes")
    p.add_argument("--min-episodes", type=int, default=MIN_EPISODES_DEFAULT,
                    help="Exclude datasets with fewer than this many episodes (only when --fetch-episodes)")
    p.add_argument("--episode-fetch-delay", type=float, default=0.25,
                    help="Seconds to wait between meta/info.json fetches (default 0.25)")
    p.add_argument("--no-robot-filter", action="store_true",
                    help="When --fetch-episodes: keep all with meta (any version/robot), do not require v3.0 + so100/so101_follower")
    p.add_argument("--robot-types", type=str, default="so100_follower,so101_follower,so_follower,bi_so_follower,bi_so101_follower,xlerobot",
                    help="Comma-separated robot_type values from meta (default: so100_follower,so101_follower,so_follower,bi_so_follower)")
    p.add_argument("--since-days", type=int, default=None,
                    help="Only include datasets with last_modified within this many days (e.g. 90 for last 3 months); use with --sort last_modified and high --limit")
    p.add_argument("--min-video-size-mb", type=float, default=None,
                    help="Require meta/info.json video_files_size_in_mb > this (only when --fetch-episodes); e.g. 10")
    p.add_argument("--resume", action="store_true",
                    help="Resume: load existing CSV, skip repo_ids already in it, append only new matches")
    p.add_argument("--window-days", type=int, default=None,
                    help="Process in time windows of this many days (e.g. 7 for weekly); use with --since-days and --state. Saves state after each window.")
    p.add_argument("--one-week", action="store_true",
                    help="With --window-days 7: process only the next 7-day window then stop (for incremental runs)")
    p.add_argument("--state", type=Path, default=None,
                    help="Path to state JSON for window progress (default: <output>.state.json when --window-days set)")
    args = p.parse_args()

    robot_types = tuple(t.strip() for t in args.robot_types.split(",") if t.strip())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    existing_rows = []
    resume_skip_repo_ids = set()
    if args.resume and args.output.exists():
        existing_rows, resume_skip_repo_ids = load_existing_csv(args.output)
        print(f"Resume: {len(existing_rows)} existing rows, skipping {len(resume_skip_repo_ids)} repo_ids")
    state_path = args.state or (args.output.with_suffix(args.output.suffix + ".state.json") if args.window_days else None)

    rows = discover(
        limit=args.limit,
        search=args.search,
        filter_lerobot=not args.no_lerobot_filter,
        sort=args.sort,
        fetch_episodes=args.fetch_episodes,
        min_episodes=args.min_episodes,
        episode_fetch_delay=args.episode_fetch_delay,
        only_target_robot=not args.no_robot_filter,
        target_robot_types=robot_types or TARGET_ROBOT_TYPES,
        since_days=args.since_days,
        min_video_size_mb=args.min_video_size_mb,
        output_path=args.output,
        existing_rows=existing_rows if args.resume else None,
        resume_skip_repo_ids=resume_skip_repo_ids if args.resume else None,
        window_days=args.window_days,
        state_path=state_path,
        one_week_only=args.one_week,
    )

    if not rows:
        print("No datasets found.")
        return

    print(f"Wrote {len(rows)} datasets to {args.output}")
    so101_count = sum(1 for r in rows if r["so101_mentioned"])
    with_task = sum(1 for r in rows if r["total_task_score"] > 0)
    eval_like_count = sum(1 for r in rows if r["is_eval_like"])
    author_has_eval_count = sum(1 for r in rows if r["author_has_eval"])
    with_episodes = sum(1 for r in rows if r.get("num_episodes") is not None)
    print(f"  SO101 mentioned: {so101_count}")
    print(f"  Task-relevant (keyword match): {with_task}")
    print(f"  Eval-like: {eval_like_count}")
    print(f"  Author has eval repo: {author_has_eval_count}")
    if args.fetch_episodes:
        print(f"  With meta (v3.0 + target robot): {with_episodes} (min_episodes>={args.min_episodes})")
        if not args.no_robot_filter:
            print(f"  Filter: codebase_version={REQUIRE_CODEBASE_V3}, robot_type in {robot_types}")
    if args.since_days:
        print(f"  Last {args.since_days} days (last_modified): {len(rows)}")
    if args.min_video_size_mb is not None:
        print(f"  video_files_size_in_mb > {args.min_video_size_mb}: {len(rows)}")
    print(f"\n  Total matching criteria: {len(rows)} datasets")
    if rows and (args.fetch_episodes or args.since_days):
        summary_path = args.output.with_suffix(args.output.suffix + ".summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Datasets matching criteria: {len(rows)}\n")
            f.write(f"With task keyword match: {with_task}\n")
            f.write("\nrepo_id\n")
            for r in rows:
                f.write(f"{r['repo_id']}\n")
        print(f"  Summary and repo list: {summary_path}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"Wrote JSON to {args.json}")


if __name__ == "__main__":
    main()
