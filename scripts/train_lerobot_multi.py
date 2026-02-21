#!/usr/bin/env python3
"""
Wrapper script to run lerobot-train with multiple local datasets.

This script:
1. Scans the datasets/ directory for downloaded datasets
2. Validates each dataset for integrity (episode counts, metadata consistency)
3. Creates symlinks in HF_LEROBOT_HOME so lerobot can find them
4. Calls lerobot-train with multi-dataset support (skipping corrupt datasets)

Usage:
    # Fresh training:
    python scripts/train_lerobot_multi.py --policy.type=pi05 [--other lerobot-train args]

    # Resume on a new machine (downloads checkpoint from HF Hub first):
    python scripts/train_lerobot_multi.py --download_checkpoint --resume=true

    # Resume locally (checkpoint already on disk):
    python scripts/train_lerobot_multi.py \\
        --config_path=outputs/train/.../train_config.json --resume=true
"""
import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
HF_LEROBOT_HOME = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache" / "huggingface" / "lerobot"))


def validate_dataset(dataset_path: Path, repo_id: str) -> tuple[bool, list[str]]:
    """
    Validate a local LeRobot dataset for integrity.

    Checks:
    1. meta/info.json exists and is valid JSON
    2. meta/episodes/ has parquet files
    3. data/ directory has parquet files
    4. Episode count in info.json matches metadata parquet row count
    5. All episode indices in data have corresponding entries in episodes metadata
    6. No gaps or out-of-range episode indices

    Returns:
        (is_valid, list_of_issues)
    """
    import pandas as pd

    issues = []

    # 1. Check meta/info.json
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return False, [f"Missing meta/info.json"]

    try:
        with open(info_path) as f:
            info = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return False, [f"Invalid meta/info.json: {e}"]

    total_episodes_info = info.get("total_episodes")
    total_frames_info = info.get("total_frames")

    if total_episodes_info is None:
        issues.append("meta/info.json missing 'total_episodes' field")
    if total_frames_info is None:
        issues.append("meta/info.json missing 'total_frames' field")

    # 2. Check meta/episodes/ parquet files
    episodes_dir = dataset_path / "meta" / "episodes"
    if not episodes_dir.exists():
        return False, issues + ["Missing meta/episodes/ directory"]

    episode_parquets = sorted(episodes_dir.rglob("*.parquet"))
    if not episode_parquets:
        return False, issues + ["No parquet files in meta/episodes/"]

    # Read all episode metadata
    try:
        ep_dfs = [pd.read_parquet(pf) for pf in episode_parquets]
        ep_meta_df = pd.concat(ep_dfs, ignore_index=True)
    except Exception as e:
        return False, issues + [f"Failed to read episode metadata: {e}"]

    meta_episode_count = len(ep_meta_df)
    if "episode_index" not in ep_meta_df.columns:
        return False, issues + ["Episode metadata missing 'episode_index' column"]

    meta_episode_indices = set(ep_meta_df["episode_index"].tolist())

    # 3. Check data/ parquet files
    data_dir = dataset_path / "data"
    if not data_dir.exists():
        # Some datasets may use videos only, but data dir should exist
        issues.append("Missing data/ directory")

    data_parquets = sorted(data_dir.rglob("*.parquet")) if data_dir.exists() else []
    if not data_parquets:
        issues.append("No parquet files in data/")

    # 4. Check episode indices in data vs metadata
    if data_parquets:
        try:
            data_episode_indices = set()
            total_data_frames = 0
            for pf in data_parquets:
                df = pd.read_parquet(pf, columns=["episode_index"])
                data_episode_indices.update(int(x) for x in df["episode_index"].unique())
                total_data_frames += len(df)
        except Exception as e:
            issues.append(f"Failed to read data parquets: {e}")
            data_episode_indices = set()
            total_data_frames = 0

        # Check for episode indices in data that are missing from metadata
        missing_in_meta = data_episode_indices - meta_episode_indices
        if missing_in_meta:
            issues.append(
                f"CRITICAL: {len(missing_in_meta)} episode(s) in data but missing from metadata: "
                f"{sorted(missing_in_meta)[:10]}{'...' if len(missing_in_meta) > 10 else ''}"
            )

        # Check for episode indices in metadata that have no data
        missing_in_data = meta_episode_indices - data_episode_indices
        if missing_in_data:
            issues.append(
                f"WARNING: {len(missing_in_data)} episode(s) in metadata but missing from data: "
                f"{sorted(missing_in_data)[:10]}{'...' if len(missing_in_data) > 10 else ''}"
            )

        # 5. Check info.json consistency
        if total_episodes_info is not None:
            max_ep_idx = max(data_episode_indices) if data_episode_indices else -1
            if meta_episode_count != total_episodes_info:
                issues.append(
                    f"Episode count mismatch: info.json says {total_episodes_info}, "
                    f"but metadata has {meta_episode_count} entries"
                )
            if len(data_episode_indices) != total_episodes_info:
                issues.append(
                    f"Data episode count mismatch: info.json says {total_episodes_info} episodes, "
                    f"but data has {len(data_episode_indices)} unique episode indices "
                    f"(max index: {max_ep_idx})"
                )

        if total_frames_info is not None and total_data_frames > 0:
            if total_data_frames != total_frames_info:
                issues.append(
                    f"Frame count mismatch: info.json says {total_frames_info}, "
                    f"but data has {total_data_frames} frames"
                )

    # Determine if critical issues exist (data/metadata mismatch = unfixable)
    critical = any("CRITICAL" in issue for issue in issues)
    return not critical, issues


def get_local_datasets() -> list[str]:
    """Scan datasets/ directory, validate each, and return list of valid repo_ids."""
    repo_ids = []
    skipped = []

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

        # Validate dataset integrity
        is_valid, issues = validate_dataset(d, repo_id)

        if issues:
            severity = "SKIPPING" if not is_valid else "WARNING (included)"
            print(f"\n  [{severity}] {repo_id}:")
            for issue in issues:
                print(f"    - {issue}")

        if is_valid:
            repo_ids.append(repo_id)
        else:
            skipped.append((repo_id, issues))

    if skipped:
        print(f"\n⚠️  Skipped {len(skipped)} corrupt dataset(s):")
        for repo_id, issues in skipped:
            print(f"    - {repo_id}: {issues[0]}")

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


def download_checkpoint(hub_repo_id: str = "tinkerbuggy/le-pickup-pi05",
                        output_dir: Path = None) -> Path:
    """
    Download a checkpoint from Hugging Face Hub for resuming training.

    Downloads the pretrained_model, training_state, and train_config.json
    into a local output directory structured for lerobot-train --resume.

    Returns the path to the train_config.json for use with --config_path.
    """
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "train" / "resumed"

    api = HfApi()

    print(f"\nDownloading checkpoint from {hub_repo_id}...")

    # List files in the repo to find checkpoints
    try:
        repo_files = api.list_repo_files(hub_repo_id)
    except Exception as e:
        print(f"ERROR: Could not access HF repo '{hub_repo_id}': {e}")
        sys.exit(1)

    # Check for train_config.json
    has_train_config = "train_config.json" in repo_files

    # Find the latest checkpoint step (look for files like pretrained_model/*, training_state/*)
    checkpoint_files = [f for f in repo_files if f.startswith(("pretrained_model/", "training_state/"))]
    top_level_model_files = [f for f in repo_files if not f.startswith((".", "training_state/")) and
                             any(f.endswith(ext) for ext in (".safetensors", ".json", ".txt", ".model"))]

    if not checkpoint_files and not top_level_model_files:
        print(f"ERROR: No model files found in {hub_repo_id}")
        sys.exit(1)

    # Download to a checkpoint directory
    checkpoint_dir = output_dir / "checkpoints" / "pretrained"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading model files to {checkpoint_dir}...")

    # Download the entire repo snapshot
    snapshot_path = snapshot_download(
        hub_repo_id,
        local_dir=str(checkpoint_dir),
        token=os.environ.get("HF_TOKEN"),
    )

    print(f"  Downloaded to: {snapshot_path}")

    # If there's a train_config.json, move it to the parent output dir
    config_path = checkpoint_dir / "train_config.json"
    output_config = output_dir / "train_config.json"

    if config_path.exists():
        shutil.copy2(str(config_path), str(output_config))
        print(f"  Config: {output_config}")
    elif has_train_config:
        # Download it separately
        hf_hub_download(
            hub_repo_id,
            filename="train_config.json",
            local_dir=str(output_dir),
            token=os.environ.get("HF_TOKEN"),
        )
        output_config = output_dir / "train_config.json"
        print(f"  Config: {output_config}")
    else:
        print("  WARNING: No train_config.json found in the repo.")
        print("           You may need to provide training arguments manually.")
        output_config = None

    # Organize into the structure lerobot expects:
    #   checkpoints/pretrained/pretrained_model/
    #   checkpoints/pretrained/training_state/
    pretrained_model_dir = checkpoint_dir / "pretrained_model"
    if not pretrained_model_dir.exists():
        # If files are at the root level, create the pretrained_model subdirectory
        pretrained_model_dir.mkdir(exist_ok=True)
        for f in checkpoint_dir.iterdir():
            if f.is_file() and f.name not in ("train_config.json",):
                shutil.move(str(f), str(pretrained_model_dir / f.name))
        print(f"  Organized model files into {pretrained_model_dir}")

    # Update the output_dir in train_config.json to point to our local path
    if output_config and output_config.exists():
        try:
            with open(output_config) as f:
                config = json.load(f)
            config["output_dir"] = str(output_dir)
            with open(output_config, "w") as f:
                json.dump(config, f, indent=2)
            print(f"  Updated output_dir in config to: {output_dir}")
        except Exception as e:
            print(f"  WARNING: Could not update config output_dir: {e}")

    print(f"\n✅ Checkpoint downloaded successfully!")
    print(f"   Output dir: {output_dir}")
    if output_config:
        print(f"   Config: {output_config}")
    print()

    return output_config


def main():
    print("=" * 60)
    print("MULTI-DATASET LEROBOT TRAINING")
    print("=" * 60)

    # Handle --download_checkpoint flag
    if "--download_checkpoint" in sys.argv:
        sys.argv.remove("--download_checkpoint")

        # Check for custom hub repo_id
        hub_repo_id = "tinkerbuggy/le-pickup-pi05"
        for arg in sys.argv[1:]:
            if arg.startswith("--hub_repo_id="):
                hub_repo_id = arg.split("=", 1)[1]
                sys.argv.remove(arg)
                break

        config_path = download_checkpoint(hub_repo_id=hub_repo_id)

        if config_path:
            # Inject --config_path into args if not already present
            arg_str = " ".join(sys.argv[1:])
            if "--config_path" not in arg_str:
                sys.argv.append(f"--config_path={config_path}")
            print(f"Continuing to training with --config_path={config_path}")
        else:
            print("WARNING: No config found. You may need to specify training args manually.")

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

    # Add defaults if not specified (skip when resuming with --config_path,
    # since the saved config already has all settings)
    arg_str = " ".join(extra_args)
    is_resuming = "--config_path" in arg_str

    if not is_resuming:
        if "--policy.type" not in arg_str:
            cmd.append("--policy.type=pi05")
        if "--policy.dtype" not in arg_str:
            cmd.append("--policy.dtype=bfloat16")
        if "--policy.gradient_checkpointing" not in arg_str:
            cmd.append("--policy.gradient_checkpointing=true")
        if "--batch_size" not in arg_str:
            cmd.append("--batch_size=4")
        if "--steps" not in arg_str:
            cmd.append("--steps=50000")
        if "--save_freq" not in arg_str:
            cmd.append("--save_freq=5000")
        if "--log_freq" not in arg_str:
            cmd.append("--log_freq=50")
        if "--policy.repo_id" not in arg_str:
            cmd.append("--policy.repo_id=tinkerbuggy/le-pickup-pi05")

    # Always disable WandB artifact upload (multi-dataset names break artifact naming)
    if "--wandb.disable_artifact" not in arg_str:
        cmd.append("--wandb.disable_artifact=true")

    print(f"\nRunning training command:")
    print(f"  {' '.join(cmd)}")
    print("=" * 60)

    # Execute
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

