# Fine Tuned Models for LeRobot SO-101 Arm

Fine-tuned **Pi0.5** and **GROOT N1.6** foundation models for **SO-101** robotic arms, trained on publicly available **LeRobot** community datasets. Target task: **pick and place**. Goal: ready-to-load models that anyone can plug into their SO-101 arm and run pick-and-place tasks.

## Current Status

| Dataset | Status |
|---------|--------|
| **15 datasets** | ✅ Downloaded |
| **542 episodes** | ✅ Curated |
| **391,885 frames** | ✅ Normalized |
| **3 cameras** | ✅ Canonical names |

## Milestones

- [x] **Discovery** - Search LeRobot HuggingFace for SO-101 datasets
- [x] **Curation** - Identify good vs bad datasets, filter by camera configs
- [x] **Download** - Fetch 15 community datasets (542 episodes)
- [x] **Normalization** - Map cameras to canonical names (`cam_overhead`, `cam_ego`, `cam_external`)
- [x] **Pipeline** - Multi-dataset loader with balanced sampling and camera masking
- [ ] **Training** - Fine-tune Pi0.5 / GROOT N1.6 on combined datasets ← *current*
- [ ] **Evaluation** - Test trained models on held-out episodes
- [ ] **Release** - Publish fine-tuned models to HuggingFace

## Project Layout

```
le-pickup/
├── scripts/
│   ├── main.py                      # Main pipeline orchestrator
│   ├── discover_lerobot_datasets.py # Search HF for SO-101 datasets
│   ├── download_datasets.py         # Download from HuggingFace
│   ├── normalize_cameras.py         # Normalize camera names
│   ├── train_multi.py               # Multi-dataset training
│   ├── multi_dataset_loader.py      # PyTorch dataset for multiple sources
│   └── camera_masking.py            # Camera dropout for robustness
├── datasets/                        # Downloaded & normalized datasets
│   ├── curated datasets for training
├── configs/                         # Generated training configs
├── data/                            # Discovery CSVs, dataset manifests
│   └── author_expansion_cameras.csv # Curated dataset list
├── requirements.txt
├── install.sh                       # Two-step installer for Pi0.5/GROOT
└── README.md
```

## Installation

### Quick Install (with Pi0.5 and GROOT support)

```bash
# Run the install script (handles two-step installation for flash-attn)
./install.sh
```

### Manual Installation

```bash
# Create virtual environment
uv venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate     # Windows

# Step 1: Install PyTorch first (required for GROOT's flash-attn)
uv pip install torch

# Step 2: Install remaining deps with --no-build-isolation
uv pip install -r requirements.txt --no-build-isolation
```

### Lightweight Install (ACT/Diffusion only, no GROOT)

```bash
uv venv .venv && source .venv/bin/activate
uv pip install lerobot pandas pyarrow tqdm
```

## Supported Models

| Model | CLI Flag | Description |
|-------|----------|-------------|
| **Pi0.5** | `--policy pi05` | Physical Intelligence VLA (default) |
| **GROOT N1.6** | `--policy groot_n1.6` | NVIDIA's latest foundation model |
| **GROOT N1.5** | `--policy groot` | NVIDIA GROOT |
| **Pi0** | `--policy pi0` | Original Pi0 |
| **ACT** | `--policy act` | Action Chunking Transformer (51.6M params) |
| **Diffusion** | `--policy diffusion` | Diffusion Policy |

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
python scripts/main.py --pipeline train --policy pi05          # Pi0.5 (default)
python scripts/main.py --pipeline train --policy groot_n1.6    # GROOT N1.6
python scripts/main.py --pipeline train --policy act           # ACT (lightweight)
```

### Pipeline stages:

1. **Download**: Fetches community datasets from Hugging Face Hub
2. **Normalize**: Maps diverse camera names to canonical format (`cam_overhead`, `cam_ego`, `cam_external`)
3. **Train**: Trains Pi0/ACT/Diffusion Policy on all datasets with task conditioning

### Training Recommendations

Based on **391,885 frames** across 15 datasets:

| Model | Epochs | Batch Size | Total Steps | Est. Time (A100) |
|-------|--------|------------|-------------|------------------|
| **Pi0.5** | 50 | 32 | ~612K | 8-12 hours |
| **GROOT N1.6** | 30 | 16 | ~734K | 10-15 hours |
| **ACT** | 100 | 64 | ~612K | 4-6 hours |

**Guidelines:**
- **Pi0.5/GROOT**: Pretrained VLMs need fewer epochs (30-50)
- **ACT**: Trains from scratch, needs more epochs (100-200)
- **Early stopping**: Save checkpoints every 5-10 epochs, evaluate on held-out episodes
- **Quick sanity check**: Run 5 epochs first to validate loss is decreasing

```bash
# Recommended training commands
python scripts/main.py --pipeline train --policy pi05 --epochs 50 --batch-size 32
python scripts/main.py --pipeline train --policy groot_n1.6 --epochs 30 --batch-size 16
python scripts/main.py --pipeline train --policy act --epochs 100 --batch-size 64
```

### For cloud training:

```bash
# Prepare locally (download + normalize)
python scripts/main.py --pipeline prepare

# Upload to cloud (datasets + scripts)
rsync -avz datasets/ scripts/ configs/ requirements.txt install.sh cloud-server:~/le-pickup/

# On cloud server
ssh cloud-server
cd ~/le-pickup
./install.sh
python scripts/main.py --pipeline train --policy pi05 --epochs 100
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

- **Good:** LeRobot v2.1/v3 format, SO101 or compatible robot, clear pick-and-place task, enough episodes and no corrupt files. See **config/dataset_criteria.yaml**.
- **Bad:** Wrong robot/format, too small, corrupt, no task description, duplicates.

## Rate limits

Discovery uses the Hugging Face Hub API (a few requests per run). No extra rate limiting is needed for normal use. Set `HF_TOKEN` for a higher quota; the client retries on 429. See **docs/RATE_LIMITS.md**.

## References

- [LeRobot datasets on Hugging Face](https://huggingface.co/datasets?other=LeRobot)
- [LeRobotDataset v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- [SO101 setup](https://huggingface.co/docs/lerobot/so101)
