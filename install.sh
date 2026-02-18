#!/bin/bash
# Install script for SO-101 Multi-Dataset Training Pipeline

set -e

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env template..."
    cat > .env << 'EOF'
# Hugging Face token (required for downloading datasets)
# Get yours at: https://huggingface.co/settings/tokens
HF_TOKEN=

# Weights & Biases token (optional, for training logging)
# Get yours at: https://wandb.ai/authorize
WANDB_API_KEY=
EOF
    echo "Created .env - please add your tokens before running the pipeline"
fi

# Load .env if it exists
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded environment from .env"
fi

echo "=== SO-101 Training Pipeline Installation ==="
echo

# Check if uv is installed, install if not
if ! command -v uv &> /dev/null; then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "uv installed successfully"
    echo
fi

# Check if we're in a virtual environment
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

# Install all requirements
echo
echo "Installing LeRobot with Pi0.5 support..."
uv pip install -r requirements.txt

echo
echo "=== Installation Complete ==="
echo
echo "Available commands:"
echo "  python scripts/main.py --status              # Check pipeline status"
echo "  python scripts/main.py --pipeline prepare   # Download + normalize"
echo "  python scripts/main.py --pipeline train --policy pi05  # Train Pi0.5"

