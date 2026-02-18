#!/bin/bash
# Install script for SO-101 Multi-Dataset Training Pipeline
# Handles the two-step installation required for GROOT's flash-attn

set -e

echo "=== SO-101 Training Pipeline Installation ==="
echo

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Creating virtual environment..."
    uv venv .venv
    source .venv/bin/activate
else
    echo "Using existing venv: $VIRTUAL_ENV"
fi

# Step 1: Install PyTorch first (required for flash-attn build)
echo
echo "Step 1/2: Installing PyTorch..."
uv pip install torch

# Step 2: Install remaining requirements with no-build-isolation
echo
echo "Step 2/2: Installing LeRobot with Pi0/GROOT support..."
uv pip install -r requirements.txt --no-build-isolation

echo
echo "=== Installation Complete ==="
echo
echo "Available commands:"
echo "  python scripts/main.py --status              # Check pipeline status"
echo "  python scripts/main.py --pipeline prepare   # Download + normalize"
echo "  python scripts/main.py --pipeline train --policy pi05        # Train Pi0.5"
echo "  python scripts/main.py --pipeline train --policy groot_n1.6  # Train GROOT N1.6"

