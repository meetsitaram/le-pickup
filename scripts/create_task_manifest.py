#!/usr/bin/env python3
"""
Create and manage task descriptions for LeRobot datasets.

This script helps create a task manifest that describes what each dataset
is doing. The manifest is used during training to provide task conditioning
(e.g., for language-conditioned policies like GROOT or Pi0).

Usage:
  # Generate initial manifest from existing datasets:
  python scripts/create_task_manifest.py --datasets-dir datasets/ --output task_manifest.json

  # Edit the generated JSON to add detailed descriptions, then use with merge:
  python scripts/merge_datasets.py --task-manifest task_manifest.json ...
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def extract_task_from_name(dataset_name: str) -> dict:
    """
    Extract likely task information from dataset name.
    Returns a dict with inferred task info.
    """
    name_lower = dataset_name.lower()
    
    # Common object patterns
    objects = []
    if "cube" in name_lower:
        objects.append("cube")
    if "block" in name_lower:
        objects.append("block")
    if "bottle" in name_lower:
        objects.append("bottle")
    if "ball" in name_lower:
        objects.append("ball")
    if "mentos" in name_lower:
        objects.append("mentos container")
    if "screwdriver" in name_lower:
        objects.append("screwdriver")
    if "plate" in name_lower:
        objects.append("plate")
    if "cup" in name_lower:
        objects.append("cup")
    if "can" in name_lower:
        objects.append("can")
    if "toy" in name_lower:
        objects.append("toy")
    
    # Common action patterns
    actions = []
    if "pick" in name_lower or "grab" in name_lower:
        actions.append("pick")
    if "place" in name_lower:
        actions.append("place")
    if "sort" in name_lower:
        actions.append("sort")
    if "stack" in name_lower:
        actions.append("stack")
    if "pour" in name_lower:
        actions.append("pour")
    if "push" in name_lower:
        actions.append("push")
    
    # Infer task type
    task_type = "pick_and_place"  # default
    if "sort" in name_lower:
        task_type = "sorting"
    elif "stack" in name_lower:
        task_type = "stacking"
    elif "pour" in name_lower:
        task_type = "pouring"
    
    return {
        "inferred_objects": objects,
        "inferred_actions": actions,
        "task_type": task_type,
    }


def load_dataset_tasks(dataset_path: Path) -> list[str]:
    """Load existing task descriptions from dataset's tasks.parquet."""
    import pandas as pd
    
    tasks_path = dataset_path / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return []
    
    try:
        df = pd.read_parquet(tasks_path)
        if "task" in df.columns:
            return df["task"].unique().tolist()
        elif "task_description" in df.columns:
            return df["task_description"].unique().tolist()
    except Exception:
        pass
    
    return []


def create_task_manifest(
    datasets_dir: Path,
    output_path: Path,
    include_existing: bool = True,
) -> dict:
    """
    Create a task manifest for all datasets in a directory.
    
    The manifest maps dataset names to task information.
    """
    manifest = {
        "_description": "Task manifest for LeRobot dataset training. "
                       "Edit the 'task_description' fields to provide detailed descriptions "
                       "of what each dataset demonstrates.",
        "_instructions": [
            "1. Review each dataset entry below",
            "2. Edit 'task_description' to be clear and specific",
            "3. Optionally add 'objects', 'actions', 'difficulty' fields",
            "4. Use this manifest with merge_datasets.py --task-manifest",
        ],
        "datasets": {},
    }
    
    datasets = sorted([
        d for d in datasets_dir.iterdir()
        if d.is_dir() and (d / "meta" / "info.json").exists()
    ])
    
    for ds_path in datasets:
        ds_name = ds_path.name
        
        # Load info.json
        with open(ds_path / "meta" / "info.json", encoding="utf-8") as f:
            info = json.load(f)
        
        # Extract task info from name
        inferred = extract_task_from_name(ds_name)
        
        # Load existing tasks if available
        existing_tasks = []
        if include_existing:
            existing_tasks = load_dataset_tasks(ds_path)
        
        # Build entry
        entry = {
            "episodes": info.get("total_episodes", 0),
            "frames": info.get("total_frames", 0),
            "robot_type": info.get("robot_type", "unknown"),
            
            # Task description - TO BE EDITED BY USER
            "task_description": existing_tasks[0] if existing_tasks else 
                f"Pick and place task with SO101 arm",
            
            # Inferred info (for reference)
            "inferred_task_type": inferred["task_type"],
            "inferred_objects": inferred["inferred_objects"],
            "inferred_actions": inferred["inferred_actions"],
            
            # Fields for user to fill
            "objects": inferred["inferred_objects"] or ["<specify objects>"],
            "actions": inferred["inferred_actions"] or ["pick", "place"],
            "difficulty": "medium",  # easy, medium, hard
            "environment": "tabletop",  # tabletop, shelf, etc.
            
            # Original task descriptions from dataset
            "original_tasks": existing_tasks,
        }
        
        manifest["datasets"][ds_name] = entry
    
    # Write manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    
    return manifest


def generate_training_prompts(manifest: dict) -> dict[str, str]:
    """
    Generate natural language prompts for each dataset.
    These can be used as task conditioning during training.
    """
    prompts = {}
    
    for ds_name, info in manifest.get("datasets", {}).items():
        task_desc = info.get("task_description", "")
        objects = info.get("objects", [])
        actions = info.get("actions", [])
        
        # Generate multiple prompt variations
        variations = []
        
        # Direct description
        variations.append(task_desc)
        
        # Object-focused
        if objects and objects[0] != "<specify objects>":
            obj_str = " and ".join(objects[:2])
            variations.append(f"Pick up the {obj_str}")
            variations.append(f"Grab the {obj_str} and place it down")
        
        # Action-focused
        if "sort" in actions:
            variations.append("Sort the objects by type")
        if "stack" in actions:
            variations.append("Stack the objects on top of each other")
        
        prompts[ds_name] = {
            "primary": task_desc,
            "variations": variations,
        }
    
    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Create task manifest for LeRobot datasets"
    )
    parser.add_argument(
        "--datasets-dir", type=Path, default=Path("datasets"),
        help="Directory containing datasets"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("config/task_manifest.json"),
        help="Output manifest file path"
    )
    parser.add_argument(
        "--generate-prompts", action="store_true",
        help="Also generate training prompts from manifest"
    )
    args = parser.parse_args()
    
    if not args.datasets_dir.exists():
        print(f"Error: {args.datasets_dir} does not exist")
        return
    
    print(f"Scanning datasets in {args.datasets_dir}...")
    manifest = create_task_manifest(args.datasets_dir, args.output)
    
    print(f"\nCreated manifest: {args.output}")
    print(f"Datasets: {len(manifest['datasets'])}")
    
    print("\n" + "=" * 60)
    print("DATASET TASK SUMMARY")
    print("=" * 60)
    
    for ds_name, info in manifest["datasets"].items():
        print(f"\n{ds_name}")
        print(f"  Episodes: {info['episodes']}")
        print(f"  Task: {info['task_description']}")
        print(f"  Objects: {info['objects']}")
        print(f"  Difficulty: {info['difficulty']}")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(f"""
1. Open {args.output} and edit each dataset's:
   - task_description: Clear natural language description
   - objects: List of objects being manipulated
   - difficulty: easy/medium/hard
   - environment: tabletop/shelf/drawer/etc.

2. Use with merge script:
   python scripts/merge_datasets.py \\
       --input-dir datasets/ \\
       --output-dir datasets_merged/ \\
       --task-manifest {args.output}

3. During training, use task descriptions for conditioning:
   - Language-conditioned policies (GROOT, RT-2, Pi0)
   - Task embedding lookup
   - Multi-task learning with task tokens
""")
    
    if args.generate_prompts:
        prompts = generate_training_prompts(manifest)
        prompts_path = args.output.with_suffix(".prompts.json")
        with open(prompts_path, "w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=2)
        print(f"\nGenerated prompts: {prompts_path}")


if __name__ == "__main__":
    main()


