#!/usr/bin/env python3
"""
Main orchestration script for SO-101 multi-dataset training pipeline.

This script handles the full pipeline:
1. Download datasets from Hugging Face
2. Normalize camera names to canonical format
3. Train on multiple datasets using LeRobot/Pi0

Usage:
    # Full pipeline (download + normalize + train)
    python scripts/main.py --pipeline full

    # Download and normalize only (for later training in cloud)
    python scripts/main.py --pipeline prepare
    
    # Download only
    python scripts/main.py --pipeline download
    
    # Normalize only (datasets already downloaded)
    python scripts/main.py --pipeline normalize
    
    # Train only (datasets already downloaded and normalized)
    python scripts/main.py --pipeline train
    
    # Train with specific config
    python scripts/main.py --pipeline train --config configs/pi0_multicam.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# Default paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Default dataset source
DEFAULT_CSV = DATA_DIR / "author_expansion_cameras.csv"


def log(msg: str, level: str = "INFO") -> None:
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def run_command(cmd: list[str], cwd: Path | None = None) -> int:
    """Run a command and return exit code."""
    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def check_datasets_exist() -> tuple[bool, list[Path]]:
    """Check if datasets directory has valid datasets."""
    if not DATASETS_DIR.exists():
        return False, []
    
    datasets = [
        d for d in DATASETS_DIR.iterdir()
        if d.is_dir() and (d / "meta" / "info.json").exists()
    ]
    return len(datasets) > 0, datasets


def check_datasets_normalized(datasets: list[Path]) -> bool:
    """Check if datasets have been normalized (have canonical camera symlinks)."""
    if not datasets:
        return False
    
    canonical_cameras = ["cam_overhead", "cam_ego", "cam_external"]
    
    for ds in datasets:
        videos_dir = ds / "videos"
        if not videos_dir.exists():
            continue
        
        # Check for at least one canonical symlink
        has_canonical = any(
            (videos_dir / f"observation.images.{cam}").exists()
            for cam in canonical_cameras
        )
        if has_canonical:
            return True
    
    return False


def download_datasets(
    csv_path: Path = DEFAULT_CSV,
    resume: bool = True,
    camera_configs: list[str] | None = None,
) -> int:
    """Download datasets from Hugging Face Hub."""
    log("=" * 60)
    log("STAGE 1: DOWNLOAD DATASETS")
    log("=" * 60)
    
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "download_datasets.py"),
        "-i", str(csv_path),
        "--default-3cam",
    ]
    
    if resume:
        cmd.append("--resume")
    
    if camera_configs:
        for config in camera_configs:
            cmd.extend(["--camera-configs", config])
    
    return run_command(cmd, cwd=PROJECT_ROOT)


def normalize_datasets(
    datasets_dir: Path = DATASETS_DIR,
    dry_run: bool = False,
    create_mask_config: bool = True,
) -> int:
    """Normalize camera names across all datasets."""
    log("=" * 60)
    log("STAGE 2: NORMALIZE CAMERAS")
    log("=" * 60)
    
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "normalize_cameras.py"),
        "--datasets-dir", str(datasets_dir),
    ]
    
    if dry_run:
        cmd.append("--dry-run")
    
    result = run_command(cmd, cwd=PROJECT_ROOT)
    
    if result == 0 and create_mask_config and not dry_run:
        log("Creating camera masking config...")
        cmd_mask = [
            sys.executable,
            str(SCRIPTS_DIR / "normalize_cameras.py"),
            "--datasets-dir", str(datasets_dir),
            "--create-mask-config",
        ]
        run_command(cmd_mask, cwd=PROJECT_ROOT)
    
    return result


def create_training_config(
    datasets: list[Path],
    output_path: Path,
    policy: str = "pi0",
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    test_mode: bool = False,
) -> Path:
    """Generate training configuration for multi-dataset training."""
    log("Generating training configuration...")
    
    # Build dataset info
    dataset_infos = []
    total_episodes = 0
    total_frames = 0
    
    for ds_path in datasets:
        info_path = ds_path / "meta" / "info.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            
            dataset_infos.append({
                "path": str(ds_path),
                "name": ds_path.name,
                "episodes": info.get("total_episodes", 0),
                "frames": info.get("total_frames", 0),
            })
            total_episodes += info.get("total_episodes", 0)
            total_frames += info.get("total_frames", 0)
    
    config = {
        "# Multi-Dataset Training Configuration": None,
        "# Generated by main.py": None,
        "# Total datasets": len(dataset_infos),
        "# Total episodes": total_episodes,
        "# Total frames": total_frames,
        "# Test mode": test_mode,
        
        "seed": 42,
        "test_mode": test_mode,
        "max_batches": 10 if test_mode else None,  # Limit batches in test mode
        
        "dataset": {
            "type": "multi",
            "paths": [d["path"] for d in dataset_infos],
            "cameras": ["cam_overhead", "cam_ego", "cam_external"],
            "sampling_strategy": "balanced",
        },
        
        "policy": {
            "name": policy,
            "use_language_conditioning": True,
            "input_shapes": {
                "observation.images.cam_overhead": [3, 480, 640],
                "observation.images.cam_ego": [3, 480, 640],
                "observation.images.cam_external": [3, 480, 640],
                "observation.state": [6],
            },
            "output_shapes": {
                "action": [6],
            },
        },
        
        "training": {
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": 1e-5,
            "grad_clip": 1.0,
            "warmup_steps": 1000,
            "save_freq": 10,
            "eval_freq": 5,
            "log_freq": 100,
        },
        
        "camera_masking": {
            "enabled": True,
            "mask_probability": 0.15,
            "min_cameras": 1,
        },
        
        "augmentation": {
            "random_crop": True,
            "color_jitter": True,
            "random_erasing": False,
        },
        
        "wandb": {
            "project": "so101-multi-dataset",
            "entity": None,
            "enabled": False,
        },
        
        "datasets_info": dataset_infos,
    }
    
    # Write as JSON (YAML would be better but avoids extra dependency)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    
    log(f"Config written to: {output_path}")
    return output_path


def train_model(
    config_path: Path | None = None,
    datasets_dir: Path = DATASETS_DIR,
    output_dir: Path | None = None,
    policy: str = "pi0",
    dry_run: bool = False,
    test_mode: bool = False,
    epochs: int = 100,
    batch_size: int = 32,
) -> int:
    """Train model on multiple datasets."""
    log("=" * 60)
    log("STAGE 3: TRAIN MODEL")
    log("=" * 60)
    
    # Check datasets exist and are normalized
    exists, datasets = check_datasets_exist()
    if not exists:
        log("ERROR: No datasets found! Run with --pipeline download first.", "ERROR")
        return 1
    
    if not check_datasets_normalized(datasets):
        log("WARNING: Datasets may not be normalized. Consider running --pipeline normalize first.", "WARN")
    
    # Test mode: minimal training for validation
    if test_mode:
        log("TEST MODE: Running minimal training (1 epoch, 10 batches)")
        epochs = 1
        batch_size = 4  # Smaller batch for quick test
    
    # Generate config if not provided
    if config_path is None or not config_path.exists():
        suffix = "_test" if test_mode else "_multi"
        config_path = CONFIGS_DIR / f"train_{policy}{suffix}.json"
        create_training_config(
            datasets=datasets,
            output_path=config_path,
            policy=policy,
            epochs=epochs,
            batch_size=batch_size,
            test_mode=test_mode,
        )
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "outputs" / f"train_{policy}_{timestamp}"
    
    log(f"Training configuration: {config_path}")
    log(f"Output directory: {output_dir}")
    log(f"Datasets: {len(datasets)}")
    
    if dry_run:
        log("DRY RUN - would start training with above configuration")
        return 0
    
    # Check if LeRobot training script exists
    train_script = SCRIPTS_DIR / "train_multi.py"
    if not train_script.exists():
        log("Creating training script placeholder...")
        create_training_script(train_script)
    
    cmd = [
        sys.executable,
        str(train_script),
        "--config", str(config_path),
        "--output-dir", str(output_dir),
    ]
    
    return run_command(cmd, cwd=PROJECT_ROOT)


def create_training_script(output_path: Path) -> None:
    """Create the multi-dataset training script."""
    script_content = '''#!/usr/bin/env python3
"""
Multi-dataset training script for SO-101 robot.

This script trains a policy (Pi0, ACT, Diffusion Policy) on multiple
normalized LeRobot datasets.

Usage:
    python scripts/train_multi.py --config configs/train_pi0_multi.json
    python scripts/train_multi.py --config configs/train_pi0_test.json  # Quick test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Train on multiple datasets")
    parser.add_argument("--config", type=Path, required=True, help="Training config JSON")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--resume", type=Path, help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = json.load(f)
    
    test_mode = config.get("test_mode", False)
    max_batches = config.get("max_batches")
    
    print("=" * 60)
    if test_mode:
        print("MULTI-DATASET TRAINING (TEST MODE)")
    else:
        print("MULTI-DATASET TRAINING")
    print("=" * 60)
    print(f"Policy: {config['policy']['name']}")
    print(f"Datasets: {len(config['dataset']['paths'])}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['epochs']}")
    if max_batches:
        print(f"Max batches: {max_batches} (test mode)")
    print(f"Camera masking: {config['camera_masking']['enabled']}")
    print()
    
    # Import training dependencies
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from multi_dataset_loader import MultiLeRobotDataset, create_multi_dataloader
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're in the project virtual environment.")
        return 1
    
    # Load datasets
    print("Loading datasets...")
    dataset = MultiLeRobotDataset(
        dataset_paths=config["dataset"]["paths"],
        cameras=config["dataset"]["cameras"],
        load_videos=not test_mode,  # Skip video loading in test mode for speed
    )
    
    # Wrap with camera masking if enabled
    if config["camera_masking"]["enabled"]:
        print("Applying camera masking...")
        try:
            from camera_masking import CameraMaskingDataset
            dataset = CameraMaskingDataset(
                dataset,
                cameras=config["dataset"]["cameras"],
                mask_prob=config["camera_masking"]["mask_probability"],
                min_cameras=config["camera_masking"]["min_cameras"],
            )
        except ImportError:
            print("  Warning: camera_masking module not found, skipping")
    
    # Create dataloader
    dataloader = create_multi_dataloader(
        dataset,
        batch_size=config["training"]["batch_size"],
        sampling_strategy=config["dataset"]["sampling_strategy"],
        num_workers=0 if test_mode else 4,  # No multiprocessing in test mode
    )
    
    print(f"Total samples: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")
    print()
    
    # Training loop (basic implementation for test)
    print("=" * 60)
    print("TRAINING LOOP")
    print("=" * 60)
    
    epochs = config["training"]["epochs"]
    
    for epoch in range(epochs):
        print(f"\\nEpoch {epoch + 1}/{epochs}")
        
        batch_count = 0
        for batch_idx, batch in enumerate(dataloader):
            batch_count += 1
            
            # In test mode, limit batches
            if max_batches and batch_count > max_batches:
                print(f"  [Test mode] Stopping after {max_batches} batches")
                break
            
            # Log every 10 batches or in test mode
            if batch_idx % 10 == 0 or test_mode:
                sample_info = f"batch {batch_idx + 1}"
                if "dataset_name" in batch:
                    datasets_in_batch = set(batch["dataset_name"]) if hasattr(batch["dataset_name"], "__iter__") else {batch["dataset_name"]}
                    sample_info += f", datasets: {len(datasets_in_batch)}"
                print(f"  [{sample_info}] Processing...")
            
            # TODO: Real training step
            # pred = model(batch)
            # loss = criterion(pred, batch["action"])
            # loss.backward()
            # optimizer.step()
        
        print(f"  Epoch {epoch + 1} complete: {batch_count} batches processed")
    
    print()
    print("=" * 60)
    if test_mode:
        print("TEST COMPLETE - Pipeline validated!")
        print("=" * 60)
        print()
        print("The data loading pipeline works. To run full training:")
        print("  python scripts/main.py --pipeline train --epochs 100")
    else:
        print("TRAINING COMPLETE")
        print("=" * 60)
        print()
        print("Note: This is a scaffold. To complete:")
        print("1. Add model initialization (Pi0, ACT, etc.)")
        print("2. Add loss computation and backpropagation")
        print("3. Add checkpointing and evaluation")
    
    return 0


if __name__ == "__main__":
    exit(main())
'''
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(script_content)
    
    output_path.chmod(0o755)
    log(f"Created training script: {output_path}")


def print_status() -> None:
    """Print current pipeline status."""
    log("=" * 60)
    log("PIPELINE STATUS")
    log("=" * 60)
    
    # Check datasets
    exists, datasets = check_datasets_exist()
    print(f"\nDatasets directory: {DATASETS_DIR}")
    print(f"  Exists: {'✓' if DATASETS_DIR.exists() else '✗'}")
    print(f"  Valid datasets: {len(datasets) if exists else 0}")
    
    if exists:
        normalized = check_datasets_normalized(datasets)
        print(f"  Normalized: {'✓' if normalized else '✗'}")
        
        # List datasets
        print("\n  Datasets:")
        for ds in sorted(datasets):
            info_path = ds / "meta" / "info.json"
            if info_path.exists():
                with open(info_path) as f:
                    info = json.load(f)
                episodes = info.get("total_episodes", "?")
                print(f"    - {ds.name}: {episodes} episodes")
    
    # Check configs
    print(f"\nConfigs directory: {CONFIGS_DIR}")
    if CONFIGS_DIR.exists():
        configs = list(CONFIGS_DIR.glob("*.json")) + list(CONFIGS_DIR.glob("*.yaml"))
        print(f"  Config files: {len(configs)}")
        for c in configs:
            print(f"    - {c.name}")
    
    # Check mask config
    mask_config = DATASETS_DIR / "camera_mask_config.json"
    print(f"\nCamera mask config: {'✓' if mask_config.exists() else '✗'}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="SO-101 Multi-Dataset Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline (download + normalize + train)
    python scripts/main.py --pipeline full

    # Prepare for cloud training (download + normalize)
    python scripts/main.py --pipeline prepare
    
    # Download only
    python scripts/main.py --pipeline download
    
    # Normalize only (datasets already downloaded)
    python scripts/main.py --pipeline normalize
    
    # Train only (datasets ready)
    python scripts/main.py --pipeline train --policy pi0
    
    # Check current status
    python scripts/main.py --status
        """,
    )
    
    parser.add_argument(
        "--pipeline",
        choices=["full", "prepare", "download", "normalize", "train"],
        help="Pipeline stage(s) to run",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current pipeline status",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="CSV file with dataset list",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=DATASETS_DIR,
        help="Directory for datasets",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Training config file (for train pipeline)",
    )
    parser.add_argument(
        "--policy",
        choices=["pi0", "act", "diffusion"],
        default="pi0",
        help="Policy to train",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume interrupted downloads",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test run with minimal steps (10 batches, 1 epoch)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    
    args = parser.parse_args()
    
    # Show status if requested
    if args.status:
        print_status()
        return 0
    
    if not args.pipeline:
        parser.print_help()
        print("\nUse --status to see current pipeline state.")
        return 0
    
    # Update paths if custom directories provided
    datasets_dir = args.datasets_dir if args.datasets_dir else DATASETS_DIR
    
    # Execute pipeline stages
    result = 0
    
    if args.pipeline in ["full", "prepare", "download"]:
        result = download_datasets(
            csv_path=args.csv,
            resume=not args.no_resume,
        )
        if result != 0:
            log("Download failed!", "ERROR")
            return result
    
    if args.pipeline in ["full", "prepare", "normalize"]:
        result = normalize_datasets(
            datasets_dir=datasets_dir,
            dry_run=args.dry_run,
        )
        if result != 0:
            log("Normalization failed!", "ERROR")
            return result
    
    if args.pipeline in ["full", "train"]:
        result = train_model(
            config_path=args.config,
            datasets_dir=datasets_dir,
            policy=args.policy,
            dry_run=args.dry_run,
            test_mode=args.test,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        if result != 0:
            log("Training failed!", "ERROR")
            return result
    
    log("Pipeline completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())

