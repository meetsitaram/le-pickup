# Dataset Curation Plan

Primary milestone: **script and search** LeRobot on Hugging Face, **identify good vs bad** datasets, **manually curate** and fix inconsistencies, **enrich with tagging/labelling** before any fine-tuning.

## 1. Discovery (scripted)

- **Script:** `scripts/discover_lerobot_datasets.py`
- **Action:** Run against Hugging Face with `filter=LeRobot`; optionally `search=so101` (or `pour`, `pick`, etc.).
- **Output:** `data/discovery_results.csv` with repo_id, author, task scores, SO101 flag, `is_eval_like`, `author_has_eval`, and optionally `num_episodes` (if `--fetch-episodes`).

**Two-phase approach (recommended):**
1. Run discovery **without** `--fetch-episodes` to get a candidate list quickly.
2. Run **`scripts/pull_sample_metadata.py`** on the CSV (e.g. `-n 15` or `-n 20`) to download `meta/info.json` (and optionally repo file list) for a sample into `data/sample_meta/`.
3. Inspect `data/sample_meta/*/info.json` and `index.json` to see real structure: `total_episodes`, schema, v2.1 vs v3, and how eval vs regular repos are named.
4. Tune `config/` (eval patterns, min_episodes) if needed, then re-run discovery with `--fetch-episodes` and `--min-episodes` for the full filtered list.
- **Use:** Sort by `so101_mentioned`, `author_has_eval`, `best_task`, `total_task_score`, `downloads` to prioritize review.

## 2. Good vs bad (criteria)

Criteria live in **`config/dataset_criteria.yaml`**. Summary:

**Good:**

- LeRobot format (v2.1 or v3.0); has state, action, images and meta (e.g. `meta/info.json` for v3).
- Robot: SO101 follower or documented compatible (e.g. same DOF + gripper).
- Task: clearly pick_and_place | pour_liquids | laundry_folding (or mappable).
- Minimum size: e.g. ≥5 episodes, ≥10 frames/episode; no corrupt Parquet/MP4.

**Bad:**

- Wrong robot with no conversion path; wrong format; too small; corrupt; no task description; duplicate.

**Review (manual):**

- Confirm task matches (e.g. “pour” in name but actually pick-only).
- Confirm action space vs SO101 or document conversion.
- Note diversity (lighting, background) if from single lab.

## 3. Manual curation workflow

1. **Triage:** From discovery CSV, mark each candidate as `include` / `exclude` / `review`.
2. **Validate:** (Optional script later) For `include`/`review`: load dataset (stream or cached), check episode count, sample frames, check for corrupt shards.
3. **Fix inconsistencies:**  
   - Normalize task strings (e.g. map to `pick_and_place` | `pour_liquids` | `laundry_folding`).  
   - Fix or document schema mismatches (e.g. different state/action dims).  
   - Use LeRobot tools: `lerobot-edit-dataset` for split/merge, add/remove features; convert v2.1→v3 if needed.
4. **Enrich:**  
   - Add canonical **task tag** and optional **difficulty** / **domain** (sim/real).  
   - Add or align **task descriptions** in `meta/tasks.jsonl` (or equivalent) for conditioning.
5. **Track:** Keep a **curated manifest** (e.g. `data/curated_manifest.json`) with repo_id, task, quality notes, and any local/converted paths.

## 4. Enrichment (tagging / labelling)

- **Task tag:** One of `pick_and_place`, `pour_liquids`, `laundry_folding` (from `config/task_taxonomy.yaml`).
- **Optional:** difficulty (easy/medium/hard), domain (sim/real), arm (so101), license.
- **Where:** In dataset card (README), or in a sidecar manifest; if you push forked datasets, in `meta/tasks.jsonl` and dataset card.

## 5. After curation

- **Ready-to-load models:** Fine-tune Pi/Groot on curated, enriched datasets; produce one (or a few) models per task that anyone can plug into an SO101 arm.
- **Reproducibility:** Document which datasets (and versions) went into each model and any conversions/applied fixes.

## References

- LeRobot datasets: <https://huggingface.co/datasets?other=LeRobot>
- LeRobotDataset v3: <https://huggingface.co/docs/lerobot/lerobot-dataset-v3>
- SO101: <https://huggingface.co/docs/lerobot/so101>
- Dataset tools: <https://huggingface.co/docs/lerobot/using_dataset_tools>
