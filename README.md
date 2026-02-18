# Fine Tuned Models for lerobot so101 arm

Fine-tuned **Pi** and **Groot** foundation models for **SO101** robotic arms, trained on publicly available **LeRobot** datasets. Target tasks: **pick and place**, **pour liquids**, **laundry folding**. Goal: ready-to-load models that anyone can plug into their SO101 arm and run these tasks, reusing hundreds of hours of data from the community.

## Primary milestone (current)

Before any fine-tuning:

1. **Script and search** the LeRobot Hugging Face space for all relevant datasets.
2. **Identify good vs bad** datasets using clear criteria.
3. **Manually curate** and fix inconsistencies (task names, schema, format).
4. **Enrich** with tagging/labelling (canonical task, difficulty, domain) then proceed to training.

## Project layout

```
solo-models/
├── config/
│   ├── task_taxonomy.yaml   # Task keywords and aliases (pick/place, pour, laundry)
│   └── dataset_criteria.yaml # Good vs bad dataset criteria
├── data/                    # Discovery CSVs, curated manifest (generated)
├── docs/
│   └── CURATION_PLAN.md     # Curation workflow and enrichment
├── scripts/
│   ├── discover_lerobot_datasets.py  # Search HF for LeRobot datasets, score by SO101 + task
│   └── README.md
├── requirements.txt
└── README.md
```

## Quick start

```bash
# Create uv env and install (run from project root)
uv venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate     # Windows
uv pip install -r requirements.txt
```

## Multi-Dataset Training Pipeline

The main script `scripts/main.py` handles the full pipeline:

```bash
# Check current status
python scripts/main.py --status

# Full pipeline (download + normalize + train)
python scripts/main.py --pipeline full

# Prepare datasets for cloud training (download + normalize only)
python scripts/main.py --pipeline prepare

# Download only
python scripts/main.py --pipeline download

# Normalize cameras only (datasets already downloaded)
python scripts/main.py --pipeline normalize

# Train only (datasets already downloaded and normalized)
python scripts/main.py --pipeline train --policy pi0
```

### Pipeline stages:

1. **Download**: Fetches community datasets from Hugging Face Hub
2. **Normalize**: Maps diverse camera names to canonical format (`cam_overhead`, `cam_ego`, `cam_external`)
3. **Train**: Trains Pi0/ACT/Diffusion Policy on all datasets with task conditioning

### For cloud training:

```bash
# Prepare locally (download + normalize)
python scripts/main.py --pipeline prepare

# Upload datasets/ folder to cloud
rsync -avz datasets/ cloud-server:~/le-pickup/datasets/

# On cloud server, train only
python scripts/main.py --pipeline train --policy pi0
```

## Dataset Discovery

```bash
# Discover LeRobot datasets (writes data/discovery_results.csv)
python scripts/discover_lerobot_datasets.py

# Narrow by keyword and cap results
python scripts/discover_lerobot_datasets.py --search "so101" --limit 200 --output data/so101_search.csv
```

Use the CSV to triage datasets (good / bad / review), then follow **docs/CURATION_PLAN.md** for manual curation and enrichment.

## Criteria (summary)

- **Good:** LeRobot v2.1/v3 format, SO101 or compatible robot, clear task (pick/place, pour, laundry), enough episodes and no corrupt files. See **config/dataset_criteria.yaml**.
- **Bad:** Wrong robot/format, too small, corrupt, no task description, duplicates.

## Rate limits

Discovery uses the Hugging Face Hub API (a few requests per run). No extra rate limiting is needed for normal use. Set `HF_TOKEN` for a higher quota; the client retries on 429. See **docs/RATE_LIMITS.md**.

## References

- [LeRobot datasets on Hugging Face](https://huggingface.co/datasets?other=LeRobot)
- [LeRobotDataset v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- [SO101 setup](https://huggingface.co/docs/lerobot/so101)
