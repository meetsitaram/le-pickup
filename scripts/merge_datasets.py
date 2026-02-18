#!/usr/bin/env python3
"""
Merge multiple LeRobot datasets into a single unified dataset.

This script combines multiple SO101 datasets with compatible schemas into
one large dataset for training. It handles:
- Episode renumbering (continuous indexing across all source datasets)
- Parquet file merging
- Video file organization (symlinks to save disk space)
- Metadata aggregation

Usage:
  # Merge all datasets in a directory:
  python scripts/merge_datasets.py --input-dir datasets/ --output-dir datasets_merged/

  # Dry run to see what would be merged:
  python scripts/merge_datasets.py --input-dir datasets/ --output-dir datasets_merged/ --dry-run

  # Merge specific datasets:
  python scripts/merge_datasets.py --datasets bluephysi01__so101_test20 bluephysi01__so101_test21 --output-dir datasets_merged/
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def load_dataset_info(dataset_path: Path) -> dict | None:
    """Load meta/info.json from a dataset."""
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return None
    with open(info_path, encoding="utf-8") as f:
        return json.load(f)


def get_dataset_episodes(dataset_path: Path) -> pd.DataFrame | None:
    """Load episode metadata from parquet files."""
    episodes_dir = dataset_path / "meta" / "episodes"
    if not episodes_dir.exists():
        return None
    
    parquet_files = sorted(episodes_dir.glob("*.parquet"))
    if not parquet_files:
        return None
    
    dfs = [pd.read_parquet(f) for f in parquet_files]
    return pd.concat(dfs, ignore_index=True) if dfs else None


def analyze_datasets(datasets: list[Path]) -> dict:
    """Analyze datasets for compatibility and gather statistics."""
    analysis = {
        "datasets": [],
        "total_episodes": 0,
        "total_frames": 0,
        "schemas_match": True,
        "features": None,
        "fps_values": set(),
        "robot_types": set(),
    }
    
    for ds_path in datasets:
        info = load_dataset_info(ds_path)
        if not info:
            continue
        
        ds_info = {
            "name": ds_path.name,
            "path": str(ds_path),
            "episodes": info.get("total_episodes", 0),
            "frames": info.get("total_frames", 0),
            "fps": info.get("fps", 30),
            "robot_type": info.get("robot_type", "unknown"),
            "features": set(info.get("features", {}).keys()),
        }
        
        analysis["datasets"].append(ds_info)
        analysis["total_episodes"] += ds_info["episodes"]
        analysis["total_frames"] += ds_info["frames"]
        analysis["fps_values"].add(ds_info["fps"])
        analysis["robot_types"].add(ds_info["robot_type"])
        
        # Check schema compatibility
        if analysis["features"] is None:
            analysis["features"] = ds_info["features"]
        elif ds_info["features"] != analysis["features"]:
            # Find common features
            analysis["features"] = analysis["features"] & ds_info["features"]
            analysis["schemas_match"] = False
    
    return analysis


def merge_datasets(
    datasets: list[Path],
    output_dir: Path,
    dry_run: bool = False,
    use_symlinks: bool = True,
    task_manifest: dict | None = None,
) -> dict:
    """
    Merge multiple LeRobot datasets into one.
    
    Args:
        datasets: List of dataset paths to merge
        output_dir: Output directory for merged dataset
        dry_run: If True, only analyze without creating files
        use_symlinks: If True, use symlinks for videos (saves disk space)
    
    Returns:
        Merge statistics
    """
    analysis = analyze_datasets(datasets)
    
    result = {
        "source_datasets": len(analysis["datasets"]),
        "total_episodes": analysis["total_episodes"],
        "total_frames": analysis["total_frames"],
        "schemas_match": analysis["schemas_match"],
        "common_features": list(analysis["features"]) if analysis["features"] else [],
        "dry_run": dry_run,
    }
    
    if dry_run:
        result["datasets"] = analysis["datasets"]
        return result
    
    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)
    (output_dir / "videos").mkdir(exist_ok=True)
    (output_dir / "meta").mkdir(exist_ok=True)
    (output_dir / "meta" / "episodes").mkdir(exist_ok=True)
    
    # Track episode renumbering
    episode_offset = 0
    frame_offset = 0
    all_data_frames = []
    all_episode_frames = []
    all_tasks = []
    merged_info = None
    
    task_idx_counter = 0
    for ds_info in analysis["datasets"]:
        ds_path = Path(ds_info["path"])
        info = load_dataset_info(ds_path)
        
        if merged_info is None:
            # Use first dataset's info as base
            merged_info = info.copy()
        
        num_episodes = ds_info["episodes"]
        num_frames = ds_info["frames"]
        
        # Store episode start for task registry
        ds_info["episode_start"] = episode_offset
        
        print(f"Processing {ds_info['name']} ({num_episodes} episodes, {num_frames} frames)...")
        
        # 1. Process data parquet files
        data_dir = ds_path / "data"
        for parquet_file in sorted(data_dir.rglob("*.parquet")):
            df = pd.read_parquet(parquet_file)
            
            # Renumber episodes and frames
            if "episode_index" in df.columns:
                df["episode_index"] = df["episode_index"] + episode_offset
            if "index" in df.columns:
                df["index"] = df["index"] + frame_offset
            
            # Add task index for this dataset
            df["task_index"] = task_idx_counter
            
            # Add source dataset info (useful for debugging)
            df["_source_dataset"] = ds_info["name"]
            
            all_data_frames.append(df)
        
        task_idx_counter += 1
        
        # 2. Process episode metadata
        episodes_df = get_dataset_episodes(ds_path)
        if episodes_df is not None:
            if "episode_index" in episodes_df.columns:
                episodes_df["episode_index"] = episodes_df["episode_index"] + episode_offset
            episodes_df["_source_dataset"] = ds_info["name"]
            all_episode_frames.append(episodes_df)
        
        # 3. Process tasks
        tasks_path = ds_path / "meta" / "tasks.parquet"
        if tasks_path.exists():
            tasks_df = pd.read_parquet(tasks_path)
            tasks_df["_source_dataset"] = ds_info["name"]
            all_tasks.append(tasks_df)
        
        # 4. Link/copy video directories
        videos_dir = ds_path / "videos"
        if videos_dir.exists():
            for cam_dir in videos_dir.iterdir():
                if not cam_dir.is_dir():
                    continue
                
                # Create camera directory in output
                out_cam_dir = output_dir / "videos" / cam_dir.name
                out_cam_dir.mkdir(exist_ok=True)
                
                # Link each chunk directory with episode-offset naming
                for chunk_dir in cam_dir.iterdir():
                    if not chunk_dir.is_dir():
                        continue
                    
                    for video_file in chunk_dir.glob("*.mp4"):
                        # Create symlink or copy
                        # Name: file-{episode_index:03d}.mp4
                        # We need to parse the original episode index and add offset
                        try:
                            # Parse original index from filename
                            orig_name = video_file.stem  # e.g., "file-000"
                            parts = orig_name.split("-")
                            if len(parts) >= 2:
                                orig_idx = int(parts[-1])
                                new_idx = orig_idx + episode_offset
                                new_name = f"file-{new_idx:03d}.mp4"
                            else:
                                new_name = f"{ds_info['name']}_{video_file.name}"
                        except ValueError:
                            new_name = f"{ds_info['name']}_{video_file.name}"
                        
                        out_chunk = out_cam_dir / chunk_dir.name
                        out_chunk.mkdir(exist_ok=True)
                        out_video = out_chunk / new_name
                        
                        if not out_video.exists():
                            if use_symlinks:
                                out_video.symlink_to(video_file.resolve())
                            else:
                                shutil.copy2(video_file, out_video)
        
        # Update offsets for next dataset
        episode_offset += num_episodes
        frame_offset += num_frames
    
    # Write merged data
    print("Writing merged parquet files...")
    
    # Combine and write data
    if all_data_frames:
        merged_data = pd.concat(all_data_frames, ignore_index=True)
        # Split into chunks of 1000 episodes each
        chunk_size = 1000
        data_chunk_dir = output_dir / "data" / "chunk-000"
        data_chunk_dir.mkdir(exist_ok=True)
        merged_data.to_parquet(data_chunk_dir / "file-000.parquet", index=False)
        result["data_rows"] = len(merged_data)
    
    # Write episode metadata
    if all_episode_frames:
        merged_episodes = pd.concat(all_episode_frames, ignore_index=True)
        merged_episodes.to_parquet(
            output_dir / "meta" / "episodes" / "chunk-000.parquet",
            index=False
        )
        result["episodes_rows"] = len(merged_episodes)
    
    # Write tasks with manifest descriptions
    task_registry = []
    task_idx = 0
    
    for ds_info in analysis["datasets"]:
        ds_name = ds_info["name"]
        
        # Get task description from manifest or use default
        if task_manifest and ds_name in task_manifest.get("datasets", {}):
            manifest_entry = task_manifest["datasets"][ds_name]
            task_desc = manifest_entry.get("task_description", f"Task from {ds_name}")
            objects = manifest_entry.get("objects", [])
            difficulty = manifest_entry.get("difficulty", "medium")
            environment = manifest_entry.get("environment", "tabletop")
        else:
            task_desc = f"Pick and place task from {ds_name}"
            objects = []
            difficulty = "medium"
            environment = "tabletop"
        
        task_registry.append({
            "task_index": task_idx,
            "task": task_desc,
            "source_dataset": ds_name,
            "objects": json.dumps(objects),
            "difficulty": difficulty,
            "environment": environment,
            "episodes_start": ds_info.get("episode_start", 0),
            "episodes_count": ds_info["episodes"],
        })
        task_idx += 1
    
    # Write unified task registry
    tasks_df = pd.DataFrame(task_registry)
    tasks_df.to_parquet(output_dir / "meta" / "tasks.parquet", index=False)
    
    # Also write as JSON for easy reading
    with open(output_dir / "meta" / "task_registry.json", "w", encoding="utf-8") as f:
        json.dump(task_registry, f, indent=2)
    
    result["tasks"] = len(task_registry)
    
    # Write merged info.json
    if merged_info:
        merged_info["total_episodes"] = episode_offset
        merged_info["total_frames"] = frame_offset
        merged_info["splits"] = {"train": f"0:{episode_offset}"}
        merged_info["_merged_from"] = [ds["name"] for ds in analysis["datasets"]]
        merged_info["_merge_date"] = pd.Timestamp.now().isoformat()
        
        with open(output_dir / "meta" / "info.json", "w", encoding="utf-8") as f:
            json.dump(merged_info, f, indent=2)
    
    # Write stats.json (placeholder)
    stats = {"merged": True, "source_count": len(analysis["datasets"])}
    with open(output_dir / "meta" / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    result["status"] = "merged"
    result["output_dir"] = str(output_dir)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple LeRobot datasets into one"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("datasets"),
        help="Directory containing datasets to merge"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("datasets_merged"),
        help="Output directory for merged dataset"
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Specific dataset folder names to merge (default: all in input-dir)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Analyze datasets without merging"
    )
    parser.add_argument(
        "--copy-videos", action="store_true",
        help="Copy videos instead of symlinking (uses more disk space)"
    )
    parser.add_argument(
        "--task-manifest", type=Path, default=None,
        help="JSON file with task descriptions per dataset (from create_task_manifest.py)"
    )
    args = parser.parse_args()
    
    # Load task manifest if provided
    task_manifest = None
    if args.task_manifest and args.task_manifest.exists():
        with open(args.task_manifest, encoding="utf-8") as f:
            task_manifest = json.load(f)
        print(f"Loaded task manifest: {args.task_manifest}")
    
    # Find datasets to merge
    if args.datasets:
        datasets = [args.input_dir / name for name in args.datasets]
        datasets = [d for d in datasets if d.exists()]
    else:
        datasets = sorted([
            d for d in args.input_dir.iterdir()
            if d.is_dir() and (d / "meta" / "info.json").exists()
        ])
    
    if not datasets:
        print("No datasets found to merge.")
        return
    
    print(f"Found {len(datasets)} datasets to merge")
    print("=" * 60)
    
    # Analyze first
    analysis = analyze_datasets(datasets)
    
    print(f"\nDatasets to merge:")
    for ds in analysis["datasets"]:
        print(f"  - {ds['name']}: {ds['episodes']} episodes, {ds['frames']} frames")
    
    print(f"\nTotal: {analysis['total_episodes']} episodes, {analysis['total_frames']} frames")
    print(f"Schemas match: {analysis['schemas_match']}")
    print(f"FPS values: {analysis['fps_values']}")
    print(f"Robot types: {analysis['robot_types']}")
    
    if analysis["features"]:
        print(f"\nCommon features ({len(analysis['features'])}):")
        for feat in sorted(analysis["features"]):
            print(f"  - {feat}")
    
    if args.dry_run:
        print("\n[DRY RUN] No files created.")
        return
    
    print("\n" + "=" * 60)
    print("MERGING DATASETS")
    print("=" * 60 + "\n")
    
    result = merge_datasets(
        datasets=datasets,
        output_dir=args.output_dir,
        dry_run=False,
        use_symlinks=not args.copy_videos,
        task_manifest=task_manifest,
    )
    
    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {result['output_dir']}")
    print(f"Total episodes: {result['total_episodes']}")
    print(f"Total frames: {result['total_frames']}")
    print(f"Data rows written: {result.get('data_rows', 'N/A')}")
    
    if not args.copy_videos:
        print("\nNote: Videos are symlinked to originals. Do not delete source datasets!")


if __name__ == "__main__":
    main()

