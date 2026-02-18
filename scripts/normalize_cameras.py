#!/usr/bin/env python3
"""
Normalize camera names across heterogeneous LeRobot datasets.

Maps diverse camera names to 3 canonical semantic slots:
  - cam_overhead: Bird's eye / top-down view (top, global)
  - cam_ego: Arm/gripper-mounted view (wrist, gripper, belly)
  - cam_external: Fixed external view (side, front, bottom)

Also supports camera masking during training for robustness.

Usage:
  # Dry run to see what would change:
  python scripts/normalize_cameras.py --datasets-dir datasets/ --dry-run

  # Apply normalization:
  python scripts/normalize_cameras.py --datasets-dir datasets/

  # Normalize specific dataset:
  python scripts/normalize_cameras.py --datasets-dir datasets/ --dataset bluephysi01__so101_test20
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

# === CAMERA MAPPING CONFIGURATION ===
# Primary mapping: first choice for each camera
CAMERA_MAPPING = {
    # Overhead / top-down views
    "top": "cam_overhead",
    "global": "cam_overhead",
    "overhead": "cam_overhead",
    
    # Ego / arm-mounted views
    "wrist": "cam_ego",
    "gripper": "cam_ego",
    "belly": "cam_ego",
    
    # External fixed views
    "side": "cam_external",
    "front": "cam_external",
    "bottom": "cam_external",
    "base": "cam_external",
}

# Fallback mapping: if primary slot is taken, use this instead
# This handles conflicts intelligently (e.g., if top takes cam_overhead, global becomes cam_external)
CAMERA_FALLBACK = {
    "global": "cam_external",      # If overhead taken, global is like an external wide view
    "belly": "cam_external",       # If ego taken, belly is like an external view
    "front": "cam_ego",            # If external taken, front could be ego-ish
    "bottom": "cam_ego",           # If external taken, bottom could be ego-ish
}

# Dataset-specific camera overrides
# Some datasets have mislabeled cameras - override the mapping here
DATASET_CAMERA_OVERRIDES = {
    "mi-kicic__so101_ds2": {
        # "front" camera is actually a gripper/ego camera in this dataset
        "front": "cam_ego",
    },
}

# Canonical camera order (for consistent indexing)
CANONICAL_CAMERAS = ["cam_overhead", "cam_ego", "cam_external"]

# Priority order when multiple cameras map to the same canonical slot
# Higher priority (earlier in list) cameras get the primary slot, others use fallback
CAMERA_PRIORITY = {
    "cam_overhead": ["top", "global", "overhead"],
    "cam_ego": ["wrist", "gripper", "belly"],
    "cam_external": ["side", "front", "bottom", "base"],
}


def get_canonical_name(original: str, dataset_name: str | None = None) -> str | None:
    """Map original camera name to canonical name.
    
    Args:
        original: Camera name (e.g., "front" or "observation.images.front")
        dataset_name: Optional dataset name to check for overrides
    
    Returns:
        Canonical camera name or None if not mappable
    """
    # Handle observation.images.X format
    if original.startswith("observation.images."):
        cam_name = original.replace("observation.images.", "")
    else:
        cam_name = original
    
    # Check for dataset-specific override first
    if dataset_name and dataset_name in DATASET_CAMERA_OVERRIDES:
        overrides = DATASET_CAMERA_OVERRIDES[dataset_name]
        if cam_name.lower() in overrides:
            return overrides[cam_name.lower()]
    
    return CAMERA_MAPPING.get(cam_name.lower())


def analyze_dataset(dataset_path: Path) -> dict:
    """Analyze a dataset's camera configuration."""
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return {"error": f"No info.json found at {info_path}"}
    
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    
    features = info.get("features", {})
    cameras = {}
    
    for key in features:
        if key.startswith("observation.images."):
            cam_name = key.replace("observation.images.", "")
            canonical = get_canonical_name(cam_name)
            cameras[cam_name] = {
                "original": key,
                "canonical": f"observation.images.{canonical}" if canonical else None,
                "mapped_to": canonical,
            }
    
    return {
        "path": str(dataset_path),
        "repo_id": dataset_path.name.replace("__", "/"),
        "cameras": cameras,
        "total_episodes": info.get("total_episodes", 0),
        "fps": info.get("fps", 30),
    }


def normalize_dataset(dataset_path: Path, dry_run: bool = True) -> dict:
    """
    Normalize camera names in a dataset.
    
    Updates:
    1. meta/info.json - feature keys
    2. videos/ directory names
    3. data/*.parquet - column names (if they reference cameras)
    
    Handles conflicts where multiple cameras map to the same canonical slot
    by keeping the highest priority camera and dropping others.
    """
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return {"error": f"No info.json found", "path": str(dataset_path)}
    
    # Load info.json
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    
    features = info.get("features", {})
    
    # Get all cameras in this dataset
    all_cameras = []
    for key in list(features.keys()):
        if key.startswith("observation.images."):
            cam_name = key.replace("observation.images.", "")
            # Skip already-canonical names
            if cam_name not in CANONICAL_CAMERAS:
                all_cameras.append(cam_name)
    
    renames = {}  # old_name -> new_name
    drops = []    # cameras to drop (only if no fallback available)
    used_slots = set()  # track which canonical slots are already assigned
    
    # Extract dataset name for override lookups
    dataset_name = dataset_path.name
    
    # First pass: assign primary mappings by priority
    canonical_groups: dict[str, list[str]] = {}  # canonical -> [original_names]
    for cam_name in all_cameras:
        canonical = get_canonical_name(cam_name, dataset_name)
        if canonical:
            canonical_groups.setdefault(canonical, []).append(cam_name)
    
    # Process each canonical slot
    for canonical in CANONICAL_CAMERAS:
        cameras = canonical_groups.get(canonical, [])
        if not cameras:
            continue
            
        # Sort by priority
        priority_list = CAMERA_PRIORITY.get(canonical, cameras)
        cameras_sorted = sorted(
            cameras,
            key=lambda c: priority_list.index(c) if c in priority_list else 999
        )
        
        # Assign highest priority to this slot
        primary = cameras_sorted[0]
        old_key = f"observation.images.{primary}"
        new_key = f"observation.images.{canonical}"
        renames[old_key] = new_key
        used_slots.add(canonical)
        
        # Handle remaining cameras (conflicts) - try fallback
        for cam in cameras_sorted[1:]:
            fallback = CAMERA_FALLBACK.get(cam)
            if fallback and fallback not in used_slots:
                # Use fallback slot
                old_key = f"observation.images.{cam}"
                new_key = f"observation.images.{fallback}"
                renames[old_key] = new_key
                used_slots.add(fallback)
            else:
                # No available slot - must drop
                drops.append(f"observation.images.{cam}")
    
    if not renames and not drops:
        return {
            "path": str(dataset_path),
            "status": "no_changes",
            "message": "No cameras need renaming",
        }
    
    result = {
        "path": str(dataset_path),
        "renames": renames,
        "drops": drops,
        "dry_run": dry_run,
    }
    
    if dry_run:
        result["status"] = "would_rename"
        return result
    
    # === APPLY CHANGES ===
    
    # 1. Update info.json
    new_features = {}
    for key, value in features.items():
        if key in drops:
            # Skip dropped cameras
            continue
        new_key = renames.get(key, key)
        new_features[new_key] = value
    
    info["features"] = new_features
    
    # Backup original
    backup_path = info_path.with_suffix(".json.bak")
    if not backup_path.exists():
        shutil.copy(info_path, backup_path)
    
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)
    
    result["info_json_updated"] = True
    
    # 2. Create symlinks for video directories (preserves originals)
    # Using symlinks instead of rename/copy to save disk space and preserve originals
    videos_dir = dataset_path / "videos"
    linked_dirs = []
    dropped_dirs = []
    
    for old_key, new_key in renames.items():
        old_dir = videos_dir / old_key
        new_dir = videos_dir / new_key
        
        if old_dir.exists() and not new_dir.exists():
            # Create symlink: new_dir -> old_dir (preserves original)
            new_dir.symlink_to(old_dir.name)
            linked_dirs.append({"from": str(old_dir), "to": str(new_dir), "type": "symlink"})
        elif old_dir.exists() and new_dir.exists():
            result["warning"] = f"Conflict: {old_dir} and {new_dir} both exist"
    
    # Handle dropped cameras - just mark them, don't move
    # The originals stay in place, we just don't create canonical links
    for drop_key in drops:
        drop_dir = videos_dir / drop_key
        if drop_dir.exists():
            dropped_dirs.append({"dir": str(drop_dir), "reason": "no available canonical slot"})
    
    result["linked_dirs"] = linked_dirs
    result["dropped_dirs"] = dropped_dirs
    
    # 3. Update parquet files (column names referencing cameras)
    # Process both data/ and meta/episodes/ directories
    parquet_dirs = [dataset_path / "data", dataset_path / "meta" / "episodes"]
    updated_parquets = []
    
    for parquet_dir in parquet_dirs:
        if not parquet_dir.exists():
            continue
        for parquet_file in parquet_dir.rglob("*.parquet"):
            try:
                df = pd.read_parquet(parquet_file)
                columns_to_rename = {}
                columns_to_drop = []
                
                for col in df.columns:
                    # Check for drops
                    for drop_key in drops:
                        if col == drop_key or col.startswith(drop_key + "."):
                            columns_to_drop.append(col)
                            break
                    else:
                        # Check for renames - handle both exact match and prefix match
                        # For episodes parquet: "videos/observation.images.side/chunk_index" -> "videos/observation.images.cam_external/chunk_index"
                        # For stats: "stats/observation.images.side/min" -> "stats/observation.images.cam_external/min"
                        for old_key, new_key in renames.items():
                            if col == old_key or col.startswith(old_key + ".") or col.startswith(old_key + "/"):
                                new_col = col.replace(old_key, new_key, 1)
                                columns_to_rename[col] = new_col
                                break
                            # Also handle "videos/old_key/..." and "stats/old_key/..." patterns
                            for prefix in ["videos/", "stats/"]:
                                if col.startswith(prefix + old_key + "/"):
                                    new_col = col.replace(prefix + old_key + "/", prefix + new_key + "/", 1)
                                    columns_to_rename[col] = new_col
                                    break
                
                if columns_to_rename or columns_to_drop:
                    # Backup
                    backup = parquet_file.with_suffix(".parquet.bak")
                    if not backup.exists():
                        shutil.copy(parquet_file, backup)
                    
                    if columns_to_drop:
                        df = df.drop(columns=columns_to_drop, errors='ignore')
                    if columns_to_rename:
                        df = df.rename(columns=columns_to_rename)
                    df.to_parquet(parquet_file, index=False)
                    updated_parquets.append({
                        "file": str(parquet_file),
                        "renamed": columns_to_rename,
                        "dropped": columns_to_drop,
                    })
            except Exception as e:
                result.setdefault("parquet_errors", []).append({
                    "file": str(parquet_file),
                    "error": str(e),
                })
    
    result["updated_parquets"] = updated_parquets
    result["status"] = "normalized"
    
    return result


def create_camera_mask_config(datasets_dir: Path, output_path: Path) -> dict:
    """
    Create a camera masking configuration file for training.
    
    Camera masking randomly drops cameras during training to make
    the model robust to missing views at inference time.
    """
    config = {
        "canonical_cameras": CANONICAL_CAMERAS,
        "masking": {
            "enabled": True,
            "strategy": "random",  # random, structured, or curriculum
            "mask_probability": 0.15,  # Probability of masking each camera
            "min_cameras": 1,  # Always keep at least 1 camera
            "max_masked": 2,  # Never mask more than 2 cameras
        },
        "camera_weights": {
            # Relative importance for loss weighting
            "cam_overhead": 1.0,
            "cam_ego": 1.2,  # Slightly more important for manipulation
            "cam_external": 0.8,
        },
        "augmentation": {
            "random_crop": True,
            "color_jitter": True,
            "gaussian_blur": False,
        },
    }
    
    # Analyze datasets
    datasets = []
    for ds_path in sorted(datasets_dir.iterdir()):
        if ds_path.is_dir() and (ds_path / "meta" / "info.json").exists():
            analysis = analyze_dataset(ds_path)
            if "error" not in analysis:
                datasets.append({
                    "name": ds_path.name,
                    "cameras": list(analysis["cameras"].keys()),
                    "episodes": analysis["total_episodes"],
                })
    
    config["datasets"] = datasets
    config["total_datasets"] = len(datasets)
    config["total_episodes"] = sum(d["episodes"] for d in datasets)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
    return config


def revert_dataset(dataset_path: Path) -> dict:
    """
    Revert a dataset to its original state using backup files.
    
    SAFETY: This function NEVER deletes files. It only:
    - Restores info.json from backup
    - Renames _dropped_* directories back to original names
    - Prints warnings about manual cleanup needed
    """
    result = {"path": str(dataset_path), "reverted": [], "warnings": []}
    
    # Restore info.json from backup
    info_path = dataset_path / "meta" / "info.json"
    backup_path = info_path.with_suffix(".json.bak")
    if backup_path.exists():
        shutil.copy(backup_path, info_path)
        result["reverted"].append("info.json restored from backup")
    
    # Find and restore dropped directories
    videos_dir = dataset_path / "videos"
    if videos_dir.exists():
        for dropped_dir in videos_dir.glob("_dropped_*"):
            # Extract original name: _dropped_global -> observation.images.global
            original_name = dropped_dir.name.replace("_dropped_", "")
            original_dir = videos_dir / f"observation.images.{original_name}"
            
            if not original_dir.exists():
                dropped_dir.rename(original_dir)
                result["reverted"].append(f"restored {original_name}")
            else:
                result["warnings"].append(f"Cannot restore {original_name}: already exists")
        
        # SAFETY: Do NOT delete canonical directories
        # Just warn the user that manual cleanup may be needed
        canonical_dirs = list(videos_dir.glob("observation.images.cam_*"))
        if canonical_dirs:
            result["warnings"].append(
                f"Found {len(canonical_dirs)} canonical dirs (cam_*) - "
                "these are renamed originals, DO NOT DELETE. "
                "Re-download dataset to get fresh copy."
            )
    
    # Restore parquet backups (copy, don't delete backup)
    data_dir = dataset_path / "data"
    if data_dir.exists():
        for backup in data_dir.rglob("*.parquet.bak"):
            original = backup.with_suffix("")
            shutil.copy(backup, original)
            result["reverted"].append(f"restored {original.name}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Normalize camera names across LeRobot datasets"
    )
    parser.add_argument(
        "--datasets-dir", type=Path, default=Path("datasets"),
        help="Directory containing downloaded datasets"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Normalize only this specific dataset (folder name)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be changed without making changes"
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Only analyze datasets, don't normalize"
    )
    parser.add_argument(
        "--create-mask-config", action="store_true",
        help="Create camera masking config file for training"
    )
    parser.add_argument(
        "--revert", action="store_true",
        help="Revert datasets to original state using backup files"
    )
    args = parser.parse_args()
    
    if not args.datasets_dir.exists():
        print(f"Error: {args.datasets_dir} does not exist")
        return
    
    # Find datasets
    if args.dataset:
        datasets = [args.datasets_dir / args.dataset]
        if not datasets[0].exists():
            print(f"Error: Dataset {args.dataset} not found")
            return
    else:
        datasets = sorted([
            d for d in args.datasets_dir.iterdir()
            if d.is_dir() and (d / "meta" / "info.json").exists()
        ])
    
    print(f"Found {len(datasets)} dataset(s) in {args.datasets_dir}\n")
    
    # Revert if requested
    if args.revert:
        print("=" * 60)
        print("REVERTING TO ORIGINAL STATE")
        print("=" * 60)
        
        for ds_path in datasets:
            result = revert_dataset(ds_path)
            print(f"\n{ds_path.name}")
            if result["reverted"]:
                for item in result["reverted"]:
                    print(f"  ✓ {item}")
            else:
                print(f"  (no backups found)")
            if result.get("warnings"):
                for warn in result["warnings"]:
                    print(f"  ⚠ {warn}")
        
        print("\nRevert complete. Run without --revert to re-normalize.")
        print("NOTE: This script NEVER deletes files. Manual cleanup may be needed.")
        return
    
    # Analyze or normalize
    if args.analyze:
        print("=" * 60)
        print("CAMERA ANALYSIS")
        print("=" * 60)
        
        for ds_path in datasets:
            analysis = analyze_dataset(ds_path)
            print(f"\n{ds_path.name}")
            print(f"  Episodes: {analysis.get('total_episodes', '?')}")
            print(f"  Cameras:")
            for cam, info in analysis.get("cameras", {}).items():
                canonical = info.get("mapped_to", "UNMAPPED")
                status = "✓" if canonical else "✗"
                print(f"    {status} {cam} -> {canonical}")
    else:
        print("=" * 60)
        print("CAMERA NORMALIZATION" + (" (DRY RUN)" if args.dry_run else ""))
        print("=" * 60)
        
        for ds_path in datasets:
            result = normalize_dataset(ds_path, dry_run=args.dry_run)
            
            print(f"\n{ds_path.name}")
            if result.get("status") == "no_changes":
                print(f"  ✓ {result['message']}")
            elif result.get("status") == "would_rename":
                if result.get("renames"):
                    print(f"  Would rename:")
                    for old, new in result.get("renames", {}).items():
                        print(f"    {old} -> {new}")
                if result.get("drops"):
                    print(f"  Would drop (conflict resolution):")
                    for drop in result.get("drops", []):
                        print(f"    ✗ {drop}")
            elif result.get("status") == "normalized":
                print(f"  ✓ Normalized!")
                print(f"    Created {len(result.get('linked_dirs', []))} symlinks (originals preserved)")
                if result.get("dropped_dirs"):
                    print(f"    Skipped {len(result.get('dropped_dirs', []))} cameras (no slot available)")
                print(f"    Updated {len(result.get('updated_parquets', []))} parquet files")
                if result.get("warning"):
                    print(f"    ⚠ {result['warning']}")
            else:
                print(f"  ✗ Error: {result.get('error', 'Unknown')}")
    
    # Create masking config if requested
    if args.create_mask_config:
        config_path = args.datasets_dir / "camera_mask_config.json"
        print(f"\n{'=' * 60}")
        print("CAMERA MASKING CONFIG")
        print("=" * 60)
        
        config = create_camera_mask_config(args.datasets_dir, config_path)
        print(f"\nCreated: {config_path}")
        print(f"Total datasets: {config['total_datasets']}")
        print(f"Total episodes: {config['total_episodes']}")
        print(f"\nMasking settings:")
        print(f"  Strategy: {config['masking']['strategy']}")
        print(f"  Mask probability: {config['masking']['mask_probability']}")
        print(f"  Min cameras kept: {config['masking']['min_cameras']}")


if __name__ == "__main__":
    main()

