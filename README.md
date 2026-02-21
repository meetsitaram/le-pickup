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
├── install.sh                       # Installer script (auto-installs uv)
├── .env                             # Your API tokens (create from template)
└── README.md
```

## Prerequisites

### System Dependencies (outside venv)

These must be installed at the OS level before running `install.sh`:

| Dependency | Required? | Purpose | Install |
|------------|-----------|---------|---------|
| **FFmpeg 4–7** | ✅ Yes | Video decoding via `torchcodec` | `apt install ffmpeg` / `brew install ffmpeg` |
| **git-lfs** | ✅ Yes | Pushing models to Hugging Face Hub | `apt install git-lfs` / `brew install git-lfs` |
| **NVIDIA GPU + drivers** | ⚠️ Recommended | GPU training (CUDA 11.8+) | [NVIDIA drivers](https://www.nvidia.com/drivers) |
| **curl** | ✅ Yes | Downloading `uv` installer | Usually pre-installed |
| **Python 3.10+** | ✅ Yes | Runtime | `apt install python3.10` |

> `install.sh` will try to install FFmpeg and git-lfs automatically, but on locked-down systems you may need to install them manually first.

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg git-lfs python3.10 python3.10-venv curl
git lfs install
```

### macOS

```bash
brew install ffmpeg git-lfs python@3.10
git lfs install
```

## Installation

### Quick Install

```bash
# Run the install script (installs uv, venv, FFmpeg, git-lfs, Python packages)
./install.sh
```

### Manual Installation

```bash
# 1. Install system deps (see Prerequisites above)
# 2. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create virtual environment
uv venv .venv
source .venv/bin/activate   # Linux/Mac

# 4. Install Python packages
uv pip install -r requirements.txt
```

### Lightweight Install (ACT/Diffusion only — no Pi0.5)

```bash
uv venv .venv && source .venv/bin/activate
uv pip install lerobot pandas pyarrow tqdm
```

## Environment Setup

Create a `.env` file in the project root with your API tokens:

```bash
cat > .env << 'EOF'
# Hugging Face token (required for downloading datasets)
# Get yours at: https://huggingface.co/settings/tokens
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Weights & Biases token (optional, for training logging)
# Get yours at: https://wandb.ai/authorize
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
EOF
```

Then edit with your actual tokens:

```bash
nano .env
```

The `install.sh` script and `main.py` automatically load `.env` — no need to manually export or login.

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
- **`num_workers=0`**: Use `--num_workers=0` for multi-dataset training. With `num_workers>0`, the DataLoader may crash with `IndexError: Invalid key: 50 is out of bounds for size 50` or `RuntimeError: Trying to resize storage that is not resizable` due to memory-mapped Arrow tensors from HuggingFace datasets conflicting with shared-memory worker processes. Single-threaded loading (`num_workers=0`) avoids this and is still fast enough on GPU.

```bash
# Recommended training commands
python scripts/main.py --pipeline train --policy pi05 --epochs 50 --batch-size 32
python scripts/main.py --pipeline train --policy groot_n1.6 --epochs 30 --batch-size 16
python scripts/main.py --pipeline train --policy act --epochs 100 --batch-size 64
```

### GPU Memory Requirements

Pi0.5 is a 3B+ parameter model. VRAM requirements vary significantly depending on which parameters are trainable:

| Configuration | Trainable Params | VRAM (bs=4) | VRAM (bs=8) | VRAM (bs=16) |
|--------------|-----------------|-------------|-------------|--------------|
| All params (default) | 3.6B | ~38 GB | ~40 GB | ~44 GB |
| `freeze_vision_encoder=true` | ~2.3B | ~30 GB | ~32 GB | ~36 GB |
| `train_expert_only=true` | ~693M | ~20 GB | ~24 GB | ~29 GB |

**Understanding Pi0.5's architecture:**

Pi0.5 has three main components, each serving a different role:

| Component | Params | Role |
|-----------|--------|------|
| **SigLIP vision encoder** | ~400M | Converts camera images into visual features. Pretrained on millions of images — already knows how to see objects, colors, and spatial relationships. |
| **Gemma language model** | ~2B | Processes task instructions (e.g., "pick up the orange") and reasons about the visual scene. Pretrained on massive text/vision corpora. |
| **Gemma action expert** | ~300M | Maps vision+language features to motor commands (joint positions for the SO-101 arm). This is the robotics-specific component. |

With `train_expert_only=true`, the vision encoder and language model are frozen — they still run during the forward pass but their weights don't update. Only the action expert learns. This is the recommended approach for consumer GPUs because:

1. **Memory**: Optimizer states (AdamW momentum + variance) are only stored for 693M params instead of 3.6B, saving ~20 GB of VRAM.
2. **Quality**: The pretrained vision and language components already understand "pick up the orange and place it in the bin" out of the box. What needs task-specific training is the action expert — *how* to move the SO-101 arm to execute that instruction.
3. **Overfitting risk**: Fine-tuning the full 3.6B model on 542 episodes risks overfitting the vision/language backbone to the training data. Freezing them preserves their general capabilities.

If you previously ran full fine-tuning (e.g., 20k steps on an A100), switching to `train_expert_only=true` preserves whatever the vision encoder and language model learned during those steps while continuing to refine the action expert.

**Critical flags for Pi0.5 training:**

- **`--policy.gradient_checkpointing=true`**: **Must be set explicitly** when using `--policy.pretrained_path` (without `--config_path`). The default is `false`, which stores all intermediate activations and uses ~30 GB regardless of batch size or freezing. This flag alone is the difference between fitting in VRAM or not.
- **`--policy.train_expert_only=true`**: Freezes the SigLIP vision encoder and Gemma language model, trains only the 300M action expert. Required for GPUs with ≤32 GB VRAM.
- **`--policy.freeze_vision_encoder=true`**: Freezes only the vision encoder (~400M params), keeps the language model and action expert trainable. A middle ground — may fit on 32 GB at batch_size=4 but tight.

**Consumer GPU tips (RTX 3090/4090/5090, 24-32 GB):**

```bash
# RTX 5090 (32 GB) — train expert only
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train_lerobot_multi.py \
    --policy.pretrained_path=outputs/train/resumed/checkpoints/pretrained \
    --policy.train_expert_only=true \
    --policy.gradient_checkpointing=true \
    --batch_size=16 --save_freq=10000 --num_workers=4 \
    --dataset.video_backend=pyav

# RTX 4090/3090 (24 GB) — train expert only, smaller batch
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train_lerobot_multi.py \
    --policy.pretrained_path=outputs/train/resumed/checkpoints/pretrained \
    --policy.train_expert_only=true \
    --policy.gradient_checkpointing=true \
    --batch_size=4 --save_freq=10000 --num_workers=4 \
    --dataset.video_backend=pyav
```

- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**: Reduces CUDA memory fragmentation. Can recover 500+ MB of wasted VRAM.
- **Desktop GPU overhead**: Xorg, GNOME, and desktop apps consume 500-800 MB of VRAM. For maximum memory, train via SSH with the display manager stopped (`sudo systemctl stop gdm3`).

### Resuming Training on a New Machine

If you've pushed a checkpoint to Hugging Face Hub, you can resume on a fresh machine:

```bash
# 1. Clone the repo and install
git clone https://github.com/meetsitaram/le-pickup.git
cd le-pickup
cp .env.example .env && nano .env   # add HF_TOKEN and WANDB_API_KEY
bash install.sh

# 2. Download datasets
source .venv/bin/activate && source .env
python scripts/main.py --pipeline download
python scripts/main.py --pipeline normalize
```

#### Option A: Download checkpoint with wget (recommended)

The HuggingFace Python downloader can stall silently on large files (no read timeout).
Use `wget` for reliable downloads with automatic retry:

```bash
# 3a. Download checkpoint via wget (resilient to network stalls)
mkdir -p outputs/train/resumed/checkpoints/pretrained
wget -c --tries=0 --timeout=30 --waitretry=5 --read-timeout=30 \
    --header="Authorization: Bearer $HF_TOKEN" \
    -O outputs/train/resumed/checkpoints/pretrained/model.safetensors \
    "https://huggingface.co/tinkerbuggy/le-pickup-pi05/resolve/main/model.safetensors"

# Download remaining config/processor files
for f in config.json train_config.json policy_preprocessor.json \
         policy_preprocessor_step_2_normalizer_processor.safetensors \
         policy_postprocessor.json \
         policy_postprocessor_step_0_unnormalizer_processor.safetensors; do
    wget -c --tries=0 --timeout=30 --read-timeout=30 \
        --header="Authorization: Bearer $HF_TOKEN" \
        -O "outputs/train/resumed/checkpoints/pretrained/$f" \
        "https://huggingface.co/tinkerbuggy/le-pickup-pi05/resolve/main/$f"
done

# 4a. Resume training from downloaded checkpoint
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONUNBUFFERED=1 nohup python scripts/train_lerobot_multi.py \
    --policy.pretrained_path=outputs/train/resumed/checkpoints/pretrained \
    --policy.train_expert_only=true \
    --policy.gradient_checkpointing=true \
    --batch_size=8 --save_freq=10000 --num_workers=4 \
    --steps=130000 --dataset.video_backend=pyav \
    > train.log 2>&1 &
```

#### Option B: Use `--download_checkpoint` (A100 / cloud only)

This uses the HuggingFace Python downloader, which requires `--resume=true` and a
`training_state/` directory. Better suited for cloud GPUs with ample VRAM:

```bash
# 3b. Auto-download and resume (needs training_state in checkpoint)
nohup python scripts/train_lerobot_multi.py \
    --download_checkpoint \
    --resume=true \
    --batch_size=32 --save_freq=10000 --num_workers=0 \
    --dataset.video_backend=pyav \
    > train.log 2>&1 &
```

#### Verify checkpoint loaded correctly

Check that the initial loss is low (~0.05-0.5). A from-scratch model starts at loss ~5-10+:

```bash
grep "loss" train.log | head -3
# ✓ Good: step:50 ... loss:0.054
# ✗ Bad:  step:50 ... loss:7.230  (loaded from scratch, not checkpoint)
```

You can also specify a different HF repo:

```bash
python scripts/train_lerobot_multi.py \
    --download_checkpoint --hub_repo_id=tinkerbuggy/le-pickup-pi05 \
    --resume=true
```

Or if the checkpoint is already on disk (e.g., copied via `rsync`/`scp`):

```bash
python scripts/train_lerobot_multi.py \
    --config_path=outputs/train/.../train_config.json \
    --resume=true
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
