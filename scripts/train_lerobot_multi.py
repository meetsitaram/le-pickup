#!/usr/bin/env python3
"""
Wrapper script to run lerobot-train with multiple local datasets.

This script:
1. Scans the datasets/ directory for downloaded datasets
2. Creates symlinks in HF_LEROBOT_HOME so lerobot can find them
3. Calls lerobot-train with multi-dataset support

Usage:
    python scripts/train_lerobot_multi.py --policy.type=pi05 [--other lerobot-train args]
"""
import os
import sys
import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
HF_LEROBOT_HOME = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache" / "huggingface" / "lerobot"))


def get_local_datasets() -> list[str]:
    """Scan datasets/ directory and return list of repo_ids."""
    repo_ids = []
    if not DATASETS_DIR.exists():
        return repo_ids

    for d in sorted(DATASETS_DIR.iterdir()):
        if not d.is_dir():
            continue
        # Check if it has meta/info.json (valid lerobot dataset)
        if not (d / "meta" / "info.json").exists():
            continue
        # Convert owner__dataset to owner/dataset
        name = d.name
        if "__" in name:
            repo_id = name.replace("__", "/", 1)
        else:
            repo_id = name
        repo_ids.append(repo_id)
    return repo_ids


def create_symlinks(repo_ids: list[str]) -> None:
    """Create symlinks in HF_LEROBOT_HOME pointing to local datasets."""
    for repo_id in repo_ids:
        # Source: datasets/owner__dataset
        local_name = repo_id.replace("/", "__")
        src = DATASETS_DIR / local_name

        # Destination: ~/.cache/huggingface/lerobot/owner/dataset
        dst = HF_LEROBOT_HOME / repo_id

        if dst.exists() or dst.is_symlink():
            # Check if it's already pointing to the right place
            if dst.is_symlink() and dst.resolve() == src.resolve():
                continue
            # If it's a real directory, replace with our normalized symlink
            if dst.is_dir() and not dst.is_symlink():
                import shutil
                shutil.rmtree(str(dst))
                print(f"  Replaced stale cache: {dst}")
            # Remove stale symlink
            elif dst.is_symlink():
                dst.unlink()

        # Create parent directory
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Create symlink
        os.symlink(str(src.resolve()), str(dst))
        print(f"  Linked: {repo_id} -> {src}")


def main():
    print("=" * 60)
    print("MULTI-DATASET LEROBOT TRAINING")
    print("=" * 60)

    # Find local datasets
    repo_ids = get_local_datasets()
    if not repo_ids:
        print("ERROR: No datasets found in datasets/")
        print("Run the download pipeline first: python scripts/main.py --pipeline download")
        sys.exit(1)

    print(f"\nFound {len(repo_ids)} datasets:")
    for rid in repo_ids:
        print(f"  - {rid}")

    # Create symlinks
    print(f"\nCreating symlinks in {HF_LEROBOT_HOME}...")
    create_symlinks(repo_ids)
    print("Done.")

    # Build repo_id list as JSON for CLI
    repo_id_json = json.dumps(repo_ids)

    # Build lerobot-train command
    # Pass through any extra arguments from CLI
    extra_args = sys.argv[1:]

    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={repo_id_json}",
    ] + extra_args

    # Add defaults if not specified
    arg_str = " ".join(extra_args)
    if "--policy.type" not in arg_str:
        cmd.append("--policy.type=pi05")
    if "--batch_size" not in arg_str:
        cmd.append("--batch_size=4")
    if "--steps" not in arg_str:
        cmd.append("--steps=50000")
    if "--save_freq" not in arg_str:
        cmd.append("--save_freq=5000")
    if "--log_freq" not in arg_str:
        cmd.append("--log_freq=50")
    # Set a default repo_id for pushing to HF Hub
    if "--policy.repo_id" not in arg_str:
        cmd.append("--policy.repo_id=tinkerbuggy/le-pickup-pi05")

    print(f"\nRunning training command:")
    print(f"  {' '.join(cmd)}")
    print("=" * 60)

    # Execute
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

