#!/usr/bin/env python3
"""
Multi-dataset loader for training on multiple LeRobot datasets simultaneously.

This module provides utilities to load and combine multiple datasets at runtime
without physically merging them. Useful for:
- Training on heterogeneous datasets with different tasks
- Experimenting with dataset mixing strategies
- Weighted sampling across datasets

Usage:
    from multi_dataset_loader import MultiLeRobotDataset, create_multi_dataloader

    # Load multiple datasets
    dataset = MultiLeRobotDataset(
        dataset_paths=["datasets/ds1", "datasets/ds2", "datasets/ds3"],
        cameras=["cam_overhead", "cam_ego", "cam_external"],
    )

    # Create dataloader with balanced sampling
    dataloader = create_multi_dataloader(
        dataset,
        batch_size=32,
        sampling_strategy="balanced",  # or "proportional", "uniform"
    )
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object
    Sampler = object
    DataLoader = None
    WeightedRandomSampler = None


@dataclass
class DatasetInfo:
    """Information about a single dataset."""
    name: str
    path: Path
    total_episodes: int
    total_frames: int
    fps: int
    robot_type: str
    features: list[str]
    cameras: list[str]
    episode_start_idx: int = 0  # Global episode index start
    frame_start_idx: int = 0    # Global frame index start
    task_description: str = "pick and place"  # Task description for VLA models
    tasks: dict[int, str] = field(default_factory=dict)  # task_index -> task description


class MultiLeRobotDataset(Dataset):
    """
    Dataset that combines multiple LeRobot datasets for training.
    
    Each sample includes:
    - Images from all cameras (normalized to canonical names)
    - Robot state observations
    - Actions (optionally chunked for Pi0.5 / flow-matching models)
    - Episode and frame indices
    - Source dataset identifier
    """
    
    def __init__(
        self,
        dataset_paths: list[str | Path],
        cameras: list[str] | None = None,
        load_videos: bool = True,
        transform: Any = None,
        action_chunk_size: int | None = None,
    ):
        """
        Args:
            dataset_paths: List of paths to LeRobot datasets
            cameras: List of camera names to load (e.g., ["cam_overhead", "cam_ego"])
            load_videos: Whether to load video frames (set False for metadata-only)
            transform: Optional transform to apply to images
            action_chunk_size: If set, return this many consecutive future actions
                per sample (required for Pi0.5 / flow-matching models). Actions
                are taken from the same episode; the last frames in an episode
                are padded by repeating the final action.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for MultiLeRobotDataset")
        
        self.dataset_paths = [Path(p) for p in dataset_paths]
        self.cameras = cameras or ["cam_overhead", "cam_ego", "cam_external"]
        self.load_videos = load_videos
        self.transform = transform
        self.action_chunk_size = action_chunk_size
        
        # Load dataset info and build index
        self.datasets: list[DatasetInfo] = []
        self.total_frames = 0
        self.total_episodes = 0
        
        self._load_datasets()
        self._build_frame_index()
    
    def _load_datasets(self) -> None:
        """Load metadata from all datasets."""
        episode_offset = 0
        frame_offset = 0
        
        for ds_path in self.dataset_paths:
            info_path = ds_path / "meta" / "info.json"
            if not info_path.exists():
                print(f"Warning: Skipping {ds_path} (no info.json)")
                continue
            
            with open(info_path, encoding="utf-8") as f:
                info = json.load(f)
            
            features = list(info.get("features", {}).keys())
            cameras = [
                f.replace("observation.images.", "")
                for f in features
                if f.startswith("observation.images.")
            ]
            
            # Load task descriptions from tasks.parquet
            tasks = {}
            tasks_path = ds_path / "meta" / "tasks.parquet"
            if tasks_path.exists():
                try:
                    tasks_df = pd.read_parquet(tasks_path)
                    # Task description is in the index, task_index is the column
                    for task_desc, row in tasks_df.iterrows():
                        task_idx = row.get("task_index", 0)
                        if isinstance(task_desc, str) and task_desc.strip():
                            tasks[task_idx] = task_desc
                    if not tasks:
                        # Fallback: use dataset name as task
                        tasks[0] = f"pick and place task: {ds_path.name}"
                except Exception:
                    tasks[0] = f"pick and place task: {ds_path.name}"
            else:
                tasks[0] = f"pick and place task: {ds_path.name}"
            
            ds_info = DatasetInfo(
                name=ds_path.name,
                path=ds_path,
                total_episodes=info.get("total_episodes", 0),
                total_frames=info.get("total_frames", 0),
                fps=info.get("fps", 30),
                robot_type=info.get("robot_type", "unknown"),
                features=features,
                cameras=cameras,
                episode_start_idx=episode_offset,
                frame_start_idx=frame_offset,
                tasks=tasks,
            )
            
            self.datasets.append(ds_info)
            episode_offset += ds_info.total_episodes
            frame_offset += ds_info.total_frames
        
        self.total_episodes = episode_offset
        self.total_frames = frame_offset
        
        print(f"Loaded {len(self.datasets)} datasets:")
        for ds in self.datasets:
            print(f"  - {ds.name}: {ds.total_episodes} episodes, {ds.total_frames} frames")
        print(f"Total: {self.total_episodes} episodes, {self.total_frames} frames")
    
    def _build_frame_index(self) -> None:
        """Build index mapping global frame index to (dataset, local_frame)."""
        self._frame_to_dataset: list[tuple[int, int]] = []  # (dataset_idx, local_frame)
        self._data_cache: dict[int, pd.DataFrame] = {}  # dataset_idx -> dataframe
        # Episode boundaries per dataset: {ds_idx: {ep_idx: (start_local, end_local)}}
        self._episode_ranges: dict[int, dict[int, tuple[int, int]]] = {}
        
        for ds_idx, ds_info in enumerate(self.datasets):
            # Load parquet data for this dataset
            data_frames = []
            data_dir = ds_info.path / "data"
            
            for parquet_file in sorted(data_dir.rglob("*.parquet")):
                df = pd.read_parquet(parquet_file)
                data_frames.append(df)
            
            if data_frames:
                ds_data = pd.concat(data_frames, ignore_index=True)
                self._data_cache[ds_idx] = ds_data
                
                # Build episode boundary index (for action chunking)
                if "episode_index" in ds_data.columns:
                    ep_ranges = {}
                    for ep_idx, group in ds_data.groupby("episode_index"):
                        ep_ranges[int(ep_idx)] = (group.index[0], group.index[-1])
                    self._episode_ranges[ds_idx] = ep_ranges
                
                # Build frame index
                for local_idx in range(len(ds_data)):
                    self._frame_to_dataset.append((ds_idx, local_idx))
    
    def __len__(self) -> int:
        return len(self._frame_to_dataset)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        ds_idx, local_idx = self._frame_to_dataset[idx]
        ds_info = self.datasets[ds_idx]
        ds_data = self._data_cache[ds_idx]
        
        row = ds_data.iloc[local_idx]
        
        sample = {
            "index": idx,
            "dataset_name": ds_info.name,
            "dataset_idx": ds_idx,
            "local_index": local_idx,
        }
        
        # Extract standard fields
        for field in ["episode_index", "frame_index", "timestamp", "task_index"]:
            if field in row:
                sample[field] = row[field]
        
        # Add task description for VLA models
        task_idx = sample.get("task_index", 0)
        if isinstance(task_idx, (int, float)):
            task_idx = int(task_idx)
        else:
            task_idx = 0
        sample["task_description"] = ds_info.tasks.get(task_idx, ds_info.tasks.get(0, "pick and place"))
        
        # Extract state
        if "observation.state" in ds_data.columns:
            state = row["observation.state"]
            if isinstance(state, np.ndarray):
                sample["observation.state"] = torch.tensor(state, dtype=torch.float32)
            else:
                sample["observation.state"] = torch.tensor([state], dtype=torch.float32)
        
        # Extract action (optionally chunked for Pi0.5)
        if "action" in ds_data.columns:
            if self.action_chunk_size and self.action_chunk_size > 1:
                # Collect chunk_size consecutive actions from the same episode
                sample["action"] = self._get_action_chunk(ds_idx, local_idx, ds_data)
            else:
                action = row["action"]
                if isinstance(action, np.ndarray):
                    sample["action"] = torch.tensor(action, dtype=torch.float32)
                else:
                    sample["action"] = torch.tensor([action], dtype=torch.float32)
        
        # Load images if requested
        if self.load_videos:
            for cam in self.cameras:
                cam_key = f"observation.images.{cam}"
                if cam_key in ds_data.columns or cam in ds_info.cameras:
                    # Try to load from video file
                    # For now, just create placeholder
                    # TODO: Implement actual video frame loading
                    sample[cam_key] = self._load_frame(ds_info, local_idx, cam)
        
        return sample
    
    def _load_frame(
        self,
        ds_info: DatasetInfo,
        frame_idx: int,
        camera: str,
    ) -> torch.Tensor:
        """
        Load a video frame for a specific camera.
        
        This is a placeholder - actual implementation would use decord/cv2
        to extract frames from the video files.
        """
        # Placeholder: return zeros with Pi0.5 expected size (224x224)
        # Real implementation would:
        # 1. Find video file based on episode_index
        # 2. Seek to frame within episode
        # 3. Decode and return tensor
        # Note: Pi0.5 expects 224x224 images for correct vision token count
        return torch.zeros(3, 224, 224, dtype=torch.float32)
    
    def _get_action_chunk(
        self, ds_idx: int, local_idx: int, ds_data: pd.DataFrame
    ) -> torch.Tensor:
        """
        Collect chunk_size consecutive actions from the same episode.

        If fewer than chunk_size future actions remain in the episode,
        the last action is repeated to fill the chunk.

        Returns:
            Tensor of shape [chunk_size, action_dim]
        """
        chunk_size = self.action_chunk_size
        row = ds_data.iloc[local_idx]

        # Find episode boundaries
        ep_idx = int(row.get("episode_index", 0))
        ep_ranges = self._episode_ranges.get(ds_idx, {})
        if ep_idx in ep_ranges:
            _ep_start, ep_end = ep_ranges[ep_idx]
        else:
            ep_end = local_idx  # fallback: treat as single-frame episode

        # Collect future actions within the episode
        actions = []
        for offset in range(chunk_size):
            future_idx = local_idx + offset
            if future_idx <= ep_end:
                a = ds_data.iloc[future_idx]["action"]
            else:
                # Past end of episode: repeat last available action
                a = ds_data.iloc[ep_end]["action"]

            if isinstance(a, np.ndarray):
                actions.append(torch.tensor(a, dtype=torch.float32))
            else:
                actions.append(torch.tensor([a], dtype=torch.float32))

        return torch.stack(actions, dim=0)  # [chunk_size, action_dim]
    
    def get_dataset_weights(self, strategy: str = "balanced") -> list[float]:
        """
        Get sampling weights for each frame based on strategy.
        
        Args:
            strategy: 
                - "balanced": Equal probability per dataset
                - "proportional": Probability proportional to dataset size
                - "uniform": Equal probability per frame (default behavior)
        
        Returns:
            List of weights for each frame
        """
        if strategy == "uniform":
            return [1.0] * len(self)
        
        weights = []
        
        if strategy == "balanced":
            # Each dataset should contribute equally
            dataset_weight = {
                ds_idx: 1.0 / (ds.total_frames * len(self.datasets))
                for ds_idx, ds in enumerate(self.datasets)
            }
        elif strategy == "proportional":
            # Natural distribution
            dataset_weight = {
                ds_idx: 1.0 / len(self)
                for ds_idx, ds in enumerate(self.datasets)
            }
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        for ds_idx, _ in self._frame_to_dataset:
            weights.append(dataset_weight[ds_idx])
        
        return weights


class BalancedMultiDatasetSampler(Sampler):
    """
    Sampler that balances samples across multiple datasets.
    
    Ensures each batch contains samples from multiple datasets,
    preventing the model from overfitting to larger datasets.
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        strategy: str = "balanced",
        shuffle: bool = True,
    ):
        self.batch_size = batch_size
        self.strategy = strategy
        self.shuffle = shuffle
        
        # Handle wrapped datasets (e.g., CameraMaskingDataset)
        inner_dataset = dataset
        while hasattr(inner_dataset, 'dataset'):
            inner_dataset = inner_dataset.dataset
        
        if not hasattr(inner_dataset, '_frame_to_dataset'):
            # Fallback: treat as single dataset
            self._dataset_indices = {0: list(range(len(dataset)))}
            self._total_len = len(dataset)
        else:
            # Build per-dataset indices from MultiLeRobotDataset
            self._dataset_indices: dict[int, list[int]] = {}
            for global_idx, (ds_idx, _) in enumerate(inner_dataset._frame_to_dataset):
                self._dataset_indices.setdefault(ds_idx, []).append(global_idx)
            self._total_len = len(inner_dataset)
    
    def __len__(self) -> int:
        return self._total_len
    
    def __iter__(self) -> Iterator[int]:
        # Create shuffled indices per dataset
        dataset_iters = {}
        for ds_idx, indices in self._dataset_indices.items():
            if self.shuffle:
                indices = indices.copy()
                random.shuffle(indices)
            dataset_iters[ds_idx] = iter(indices)
        
        # Yield in round-robin fashion across datasets
        active_datasets = list(dataset_iters.keys())
        
        while active_datasets:
            for ds_idx in active_datasets[:]:
                try:
                    yield next(dataset_iters[ds_idx])
                except StopIteration:
                    active_datasets.remove(ds_idx)


def create_multi_dataloader(
    dataset: MultiLeRobotDataset,
    batch_size: int = 32,
    sampling_strategy: str = "balanced",
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for multi-dataset training.
    
    Args:
        dataset: MultiLeRobotDataset instance
        batch_size: Batch size
        sampling_strategy: "balanced", "proportional", or "uniform"
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
    
    Returns:
        PyTorch DataLoader
    """
    if sampling_strategy == "uniform":
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
    elif sampling_strategy == "balanced":
        sampler = BalancedMultiDatasetSampler(
            dataset,
            batch_size=batch_size,
            strategy="balanced",
            shuffle=shuffle,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
        )
    else:
        # Use weighted random sampling
        weights = dataset.get_dataset_weights(sampling_strategy)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
        )


# === TRAINING INTEGRATION EXAMPLE ===

TRAINING_EXAMPLE = """
# Example: Training on multiple LeRobot datasets

from multi_dataset_loader import MultiLeRobotDataset, create_multi_dataloader
from camera_masking import CameraMaskingDataset

# 1. Load multiple normalized datasets
dataset = MultiLeRobotDataset(
    dataset_paths=[
        "datasets/bluephysi01__so101_test20",
        "datasets/bluephysi01__so101_test21",
        "datasets/gannbayar__so101_sorting_3cam_480p",
        # ... add all your datasets
    ],
    cameras=["cam_overhead", "cam_ego", "cam_external"],
)

# 2. Optionally wrap with camera masking for robustness
masked_dataset = CameraMaskingDataset(
    dataset,
    cameras=["cam_overhead", "cam_ego", "cam_external"],
    mask_prob=0.15,
)

# 3. Create balanced dataloader
dataloader = create_multi_dataloader(
    masked_dataset,
    batch_size=32,
    sampling_strategy="balanced",  # Ensures all datasets contribute equally
    num_workers=4,
)

# 4. Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        images = {
            k: batch[k] for k in batch 
            if k.startswith("observation.images.")
        }
        state = batch["observation.state"]
        action = batch["action"]
        
        # Track which dataset each sample came from
        dataset_names = batch["dataset_name"]
        
        # Forward pass
        pred_action = model(images, state)
        loss = criterion(pred_action, action)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
"""


if __name__ == "__main__":
    print("Multi-Dataset Loader")
    print("=" * 60)
    print(TRAINING_EXAMPLE)
    
    # Demo with available datasets
    from pathlib import Path
    
    datasets_dir = Path("datasets")
    if datasets_dir.exists():
        dataset_paths = sorted([
            d for d in datasets_dir.iterdir()
            if d.is_dir() and (d / "meta" / "info.json").exists()
        ])
        
        if dataset_paths:
            print("\n" + "=" * 60)
            print("ANALYZING AVAILABLE DATASETS")
            print("=" * 60 + "\n")
            
            dataset = MultiLeRobotDataset(
                dataset_paths=dataset_paths,
                load_videos=False,  # Skip video loading for demo
            )
            
            print(f"\nTotal samples: {len(dataset)}")


