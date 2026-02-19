#!/bin/bash
# Install script for SO-101 Multi-Dataset Training Pipeline

set -e

echo "=== SO-101 Training Pipeline Installation ==="
echo

# ── 1. System dependencies ──────────────────────────────────────────────
echo "Checking system dependencies..."

# FFmpeg (required by torchcodec for decoding dataset videos)
if ! command -v ffmpeg &> /dev/null; then
    echo "  Installing FFmpeg (required for video decoding)..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg 2>/dev/null || \
            apt-get update -qq && apt-get install -y -qq ffmpeg
    elif command -v yum &> /dev/null; then
        sudo yum install -y ffmpeg 2>/dev/null || yum install -y ffmpeg
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y ffmpeg 2>/dev/null || dnf install -y ffmpeg
    elif command -v brew &> /dev/null; then
        brew install ffmpeg
    else
        echo "  WARNING: Could not install FFmpeg automatically."
        echo "  Please install FFmpeg manually: https://ffmpeg.org/download.html"
    fi
else
    echo "  ✓ FFmpeg $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"
fi

# git-lfs (required for pushing models to Hugging Face Hub)
if ! command -v git-lfs &> /dev/null; then
    echo "  Installing git-lfs (required for HF model uploads)..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get install -y -qq git-lfs 2>/dev/null || apt-get install -y -qq git-lfs
    elif command -v yum &> /dev/null; then
        sudo yum install -y git-lfs 2>/dev/null || yum install -y git-lfs
    elif command -v brew &> /dev/null; then
        brew install git-lfs
    else
        echo "  WARNING: Could not install git-lfs automatically."
        echo "  Please install it: https://git-lfs.com"
    fi
    git lfs install 2>/dev/null || true
else
    echo "  ✓ git-lfs $(git lfs version 2>&1 | head -1)"
fi

# NVIDIA GPU check (optional but recommended)
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    echo "  ✓ GPU: ${GPU_NAME} (driver ${CUDA_VER})"
else
    echo "  ⚠ No NVIDIA GPU detected — training will be CPU-only (very slow)"
fi

echo

# ── 2. uv package manager ───────────────────────────────────────────────
if ! command -v uv &> /dev/null; then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "uv installed successfully"
    echo
fi

# ── 3. Python virtual environment ────────────────────────────────────────
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ".venv" ]; then
        echo "Using existing .venv..."
    else
        echo "Creating virtual environment..."
        uv venv .venv
    fi
    source .venv/bin/activate
else
    echo "Using existing venv: $VIRTUAL_ENV"
fi

# ── 4. Python packages ──────────────────────────────────────────────────
echo
echo "Installing Python packages (LeRobot + Pi0.5 + dependencies)..."
uv pip install -r requirements.txt

# ── 5. Environment file ─────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    echo
    echo "Creating .env template..."
    cat > .env << 'EOF'
# Hugging Face token (required for downloading datasets and pushing models)
# Get yours at: https://huggingface.co/settings/tokens
HF_TOKEN=

# Weights & Biases token (optional, for training logging)
# Get yours at: https://wandb.ai/authorize
WANDB_API_KEY=
EOF
    echo "Created .env — please add your tokens before running the pipeline"
fi

# Load .env if it exists
if [ -f ".env" ]; then
    set -a
    source <(grep -v '^\s*#' .env | grep -v '^\s*$')
    set +a
    echo "Loaded environment from .env"
fi

# ── Done ─────────────────────────────────────────────────────────────────
echo
echo "=== Installation Complete ==="
echo
echo "Next steps:"
echo "  1. Edit .env with your HF_TOKEN (and optionally WANDB_API_KEY)"
echo "  2. Run the pipeline:"
echo
echo "  python scripts/main.py --status                              # Check status"
echo "  python scripts/main.py --pipeline prepare                    # Download + normalize"
echo "  python scripts/train_lerobot_multi.py --policy.type=pi05     # Train Pi0.5"

