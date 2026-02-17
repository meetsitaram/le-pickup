# Scripts

## discover_lerobot_datasets.py

Searches the Hugging Face Hub for datasets tagged **LeRobot** and scores them for:

- **SO101 relevance** – repo id / description contains "so101", "SO-101", etc.
- **Task relevance** – keywords for pick_and_place, pour_liquids, laundry_folding (see `config/task_taxonomy.yaml`).
- **Eval vs regular** – repos with "eval", "evaluation", "benchmark", "trained", etc. in name/description are flagged as `is_eval_like`. If an author has any eval-like repo, all their repos get `author_has_eval` (prioritize their regular data).
- **Metadata filter** – `--fetch-episodes` reads `meta/info.json`. By default keeps only datasets with **codebase_version "v3.0"** and **robot_type** in `so100_follower`, `so101_follower` (SO100/SO101 family; name may not be in dataset title). Also excludes datasets with fewer than `--min-episodes` (default 10). Use `--no-robot-filter` to keep all that have meta.

**Output:** CSV with columns including `repo_id`, `author`, `downloads`, `likes`, `so101_mentioned`, `is_eval_like`, `author_has_eval`, `num_episodes` (if fetched), `task_*`, `best_task`, etc.

**Sort (Hub search order):** `--sort downloads` (default), `--sort likes` (hearts), `--sort last_modified` (newest updated first), or `--sort created_at` (newest created first). CSV includes `last_modified` and `created_at` columns.

**Usage:**

```bash
# Default: LeRobot tag, 500 results, sorted by downloads (desc)
python scripts/discover_lerobot_datasets.py

# Sort by hearts (likes) instead of downloads
python scripts/discover_lerobot_datasets.py --sort likes --limit 100

# Fetch episode counts and exclude datasets with < 10 episodes
python scripts/discover_lerobot_datasets.py --fetch-episodes --min-episodes 10

# Last 3 months, so100/so101_follower, v3.0, video > 10 MB, min 10 episodes (output + summary file)
.venv\Scripts\python scripts/discover_lerobot_datasets.py --fetch-episodes --sort last_modified --since-days 90 --min-video-size-mb 10 --limit 2000 -o data/so101_v3_recent_3months.csv

# Week-by-week: process 7-day windows from now back to 90 days ago (saves state after each week; resume skips existing repo_ids)
.venv\Scripts\python scripts/discover_lerobot_datasets.py --fetch-episodes --sort last_modified --since-days 90 --min-video-size-mb 10 --window-days 7 --limit 2000 -o data/so101_v3_recent_3months.csv

# One week only (then stop); run again with --resume to do the next week
.venv\Scripts\python scripts/discover_lerobot_datasets.py --fetch-episodes --sort last_modified --since-days 90 --min-video-size-mb 10 --window-days 7 --one-week --limit 2000 -o data/so101_v3_recent_3months.csv
.venv\Scripts\python scripts/discover_lerobot_datasets.py --resume --fetch-episodes --sort last_modified --since-days 90 --min-video-size-mb 10 --window-days 7 --one-week --limit 2000 -o data/so101_v3_recent_3months.csv

# Narrow by search
python scripts/discover_lerobot_datasets.py --search "so101" --limit 200 --output data/so101.csv
```

**Recommended:** Run discovery without `--fetch-episodes` first, then **pull_sample_metadata.py** on the CSV to download meta for a small sample; inspect `data/sample_meta/` to tune eval patterns and episode logic before re-running with `--fetch-episodes`.

Use the CSV to **identify good vs bad** and **manual curation** (see `docs/CURATION_PLAN.md`). Filter out `is_eval_like` when building training candidate lists; use `author_has_eval` to prioritize authors who also publish eval data.

---

## pull_sample_metadata.py

Pulls metadata for a **sample** of repos from a discovery CSV so you can inspect real structure before locking in heuristics.

- Downloads `meta/info.json` for each sampled repo into `data/sample_meta/<repo>/info.json`.
- Optionally `--list-files` to save repo file list.
- Writes `data/sample_meta/index.json` with repo_id, total_episodes, and any errors.

**Usage:**

```bash
# After discovery
python scripts/discover_lerobot_datasets.py --limit 50 -o data/discovery.csv
python scripts/pull_sample_metadata.py data/discovery.csv -n 15 -o data/sample_meta
# Inspect data/sample_meta/ then tune config and re-run discovery with --fetch-episodes
python scripts/pull_sample_metadata.py data/discovery.csv -n 20 --by so101_mentioned --list-files
```
