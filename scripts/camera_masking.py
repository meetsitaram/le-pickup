#!/usr/bin/env python3
"""
Camera masking utilities for robust multi-view robot learning.

Camera masking randomly drops camera inputs during training to make models
robust to missing views at inference time. This is especially useful when:
- Training on heterogeneous datasets with different camera setups
- Deploying to robots where cameras may fail or be occluded
- Testing generalization across camera configurations

Usage in training:
    from camera_masking import CameraMasker, CameraMaskingDataset

    masker = CameraMasker(
        cameras=["cam_overhead", "cam_ego", "cam_external"],
        mask_prob=0.15,
        min_cameras=1,
    )

    # During training loop:
    masked_images, mask = masker.apply(images_dict)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class CameraMaskConfig:
    """Configuration for camera masking."""
    
    # List of canonical camera names
    cameras: list[str] = field(default_factory=lambda: [
        "cam_overhead", "cam_ego", "cam_external"
    ])
    
    # Probability of masking each camera
    mask_probability: float = 0.15
    
    # Minimum number of cameras to keep (never mask all)
    min_cameras: int = 1
    
    # Maximum cameras to mask at once
    max_masked: int = 2
    
    # Masking strategy: "random", "structured", "curriculum"
    strategy: str = "random"
    
    # For curriculum learning: start with no masking, increase over training
    curriculum_start_epoch: int = 0
    curriculum_end_epoch: int = 100
    curriculum_max_prob: float = 0.3
    
    # What to replace masked cameras with
    mask_value: str = "zeros"  # "zeros", "noise", "mean", "learned"


class CameraMasker:
    """
    Apply camera masking to multi-view observations.
    
    Randomly masks (drops) camera views during training to create a model
    that's robust to missing cameras at inference time.
    """
    
    def __init__(
        self,
        cameras: list[str] | None = None,
        mask_prob: float = 0.15,
        min_cameras: int = 1,
        max_masked: int = 2,
        strategy: str = "random",
        mask_value: str = "zeros",
    ):
        self.config = CameraMaskConfig(
            cameras=cameras or ["cam_overhead", "cam_ego", "cam_external"],
            mask_probability=mask_prob,
            min_cameras=min_cameras,
            max_masked=max_masked,
            strategy=strategy,
            mask_value=mask_value,
        )
        self._current_epoch = 0
    
    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for curriculum learning."""
        self._current_epoch = epoch
    
    def get_effective_mask_prob(self) -> float:
        """Get current mask probability (may vary with curriculum)."""
        if self.config.strategy != "curriculum":
            return self.config.mask_probability
        
        # Linear curriculum from 0 to max_prob
        start = self.config.curriculum_start_epoch
        end = self.config.curriculum_end_epoch
        
        if self._current_epoch < start:
            return 0.0
        elif self._current_epoch >= end:
            return self.config.curriculum_max_prob
        else:
            progress = (self._current_epoch - start) / (end - start)
            return progress * self.config.curriculum_max_prob
    
    def generate_mask(self, available_cameras: list[str] | None = None) -> dict[str, bool]:
        """
        Generate a camera mask.
        
        Returns:
            Dict mapping camera name -> True if visible, False if masked
        """
        cameras = available_cameras or self.config.cameras
        n_cameras = len(cameras)
        
        if n_cameras <= self.config.min_cameras:
            # Can't mask any cameras
            return {cam: True for cam in cameras}
        
        mask_prob = self.get_effective_mask_prob()
        
        if self.config.strategy == "random":
            # Randomly decide which cameras to mask
            mask = {}
            n_masked = 0
            max_maskable = min(
                self.config.max_masked,
                n_cameras - self.config.min_cameras
            )
            
            for cam in cameras:
                if n_masked < max_maskable and random.random() < mask_prob:
                    mask[cam] = False
                    n_masked += 1
                else:
                    mask[cam] = True
            
            return mask
        
        elif self.config.strategy == "structured":
            # Mask entire "types" of cameras (e.g., all ego cameras)
            # Useful for testing robustness to specific camera failures
            if random.random() < mask_prob:
                # Pick a random camera to mask
                maskable = cameras[:n_cameras - self.config.min_cameras]
                if maskable:
                    to_mask = random.choice(maskable)
                    return {cam: (cam != to_mask) for cam in cameras}
            
            return {cam: True for cam in cameras}
        
        else:
            # Default: no masking
            return {cam: True for cam in cameras}
    
    def apply_mask_numpy(
        self,
        images: dict[str, np.ndarray],
        mask: dict[str, bool] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
        """
        Apply masking to numpy image arrays.
        
        Args:
            images: Dict mapping camera name -> image array (H, W, C) or (B, H, W, C)
            mask: Optional pre-generated mask. If None, generates new mask.
        
        Returns:
            Tuple of (masked_images, mask_applied)
        """
        if mask is None:
            mask = self.generate_mask(list(images.keys()))
        
        masked_images = {}
        for cam, img in images.items():
            if mask.get(cam, True):
                # Camera is visible
                masked_images[cam] = img
            else:
                # Camera is masked
                if self.config.mask_value == "zeros":
                    masked_images[cam] = np.zeros_like(img)
                elif self.config.mask_value == "noise":
                    masked_images[cam] = np.random.rand(*img.shape).astype(img.dtype)
                    if img.max() > 1:
                        masked_images[cam] *= 255
                elif self.config.mask_value == "mean":
                    masked_images[cam] = np.full_like(img, img.mean())
                else:
                    masked_images[cam] = np.zeros_like(img)
        
        return masked_images, mask
    
    def apply_mask_torch(
        self,
        images: dict[str, "torch.Tensor"],
        mask: dict[str, bool] | None = None,
    ) -> tuple[dict[str, "torch.Tensor"], dict[str, bool]]:
        """
        Apply masking to PyTorch tensors.
        
        Args:
            images: Dict mapping camera name -> tensor (B, C, H, W) or (C, H, W)
            mask: Optional pre-generated mask. If None, generates new mask.
        
        Returns:
            Tuple of (masked_images, mask_applied)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for apply_mask_torch")
        
        if mask is None:
            mask = self.generate_mask(list(images.keys()))
        
        masked_images = {}
        for cam, img in images.items():
            if mask.get(cam, True):
                masked_images[cam] = img
            else:
                if self.config.mask_value == "zeros":
                    masked_images[cam] = torch.zeros_like(img)
                elif self.config.mask_value == "noise":
                    masked_images[cam] = torch.rand_like(img)
                elif self.config.mask_value == "mean":
                    masked_images[cam] = torch.full_like(img, img.mean())
                else:
                    masked_images[cam] = torch.zeros_like(img)
        
        return masked_images, mask
    
    def apply(
        self,
        images: dict[str, Any],
        mask: dict[str, bool] | None = None,
    ) -> tuple[dict[str, Any], dict[str, bool]]:
        """
        Apply masking to images (auto-detects numpy vs torch).
        """
        if not images:
            return images, {}
        
        first_img = next(iter(images.values()))
        
        if HAS_TORCH and isinstance(first_img, torch.Tensor):
            return self.apply_mask_torch(images, mask)
        else:
            return self.apply_mask_numpy(images, mask)


class CameraMaskingDataset:
    """
    Wrapper that adds camera masking to any LeRobot-style dataset.
    
    Usage:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        
        base_dataset = LeRobotDataset("path/to/dataset")
        masked_dataset = CameraMaskingDataset(
            base_dataset,
            mask_prob=0.15,
            cameras=["cam_overhead", "cam_ego", "cam_external"],
        )
    """
    
    def __init__(
        self,
        dataset: Any,
        cameras: list[str] | None = None,
        mask_prob: float = 0.15,
        min_cameras: int = 1,
        enabled: bool = True,
    ):
        self.dataset = dataset
        self.enabled = enabled
        self.masker = CameraMasker(
            cameras=cameras,
            mask_prob=mask_prob,
            min_cameras=min_cameras,
        )
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        
        if not self.enabled:
            return item
        
        # Find image keys in the item
        image_keys = [k for k in item.keys() if k.startswith("observation.images.")]
        
        if not image_keys:
            return item
        
        # Extract images
        images = {k: item[k] for k in image_keys}
        
        # Apply masking
        masked_images, mask = self.masker.apply(images)
        
        # Update item
        for k, v in masked_images.items():
            item[k] = v
        
        # Add mask info to item (useful for loss weighting)
        item["_camera_mask"] = mask
        
        return item
    
    def set_epoch(self, epoch: int) -> None:
        """Update epoch for curriculum learning."""
        self.masker.set_epoch(epoch)


def create_mask_indicator_tensor(
    mask: dict[str, bool],
    cameras: list[str],
) -> np.ndarray:
    """
    Create a binary indicator tensor for the mask.
    
    Useful for conditioning the model on which cameras are available.
    
    Args:
        mask: Dict mapping camera name -> visibility
        cameras: Ordered list of camera names
    
    Returns:
        Binary array of shape (n_cameras,) where 1 = visible, 0 = masked
    """
    return np.array([1.0 if mask.get(cam, True) else 0.0 for cam in cameras])


# === TRAINING INTEGRATION EXAMPLE ===

TRAINING_EXAMPLE = """
# Example: Integrating camera masking with LeRobot training

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from camera_masking import CameraMaskingDataset, CameraMasker

# 1. Load your normalized dataset
dataset = LeRobotDataset(
    repo_id="local/normalized_dataset",
    root="datasets/",
)

# 2. Wrap with camera masking
masked_dataset = CameraMaskingDataset(
    dataset,
    cameras=["cam_overhead", "cam_ego", "cam_external"],
    mask_prob=0.15,
    min_cameras=1,
)

# 3. In your training loop
for epoch in range(num_epochs):
    masked_dataset.set_epoch(epoch)  # For curriculum learning
    
    for batch in dataloader:
        # batch["observation.images.cam_overhead"] may be zeros if masked
        # batch["_camera_mask"] tells you which cameras were masked
        
        images = {
            k: batch[k] for k in batch 
            if k.startswith("observation.images.")
        }
        
        # Option A: Concatenate all cameras (masked ones are zeros)
        # Option B: Use mask indicator as additional input
        mask_indicator = batch.get("_camera_mask", {})
        
        # Forward pass with mask-aware model
        actions = model(images, state=batch["observation.state"])
        
        # Compute loss (optionally weight by visible cameras)
        loss = compute_loss(actions, batch["action"])
"""


if __name__ == "__main__":
    # Demo / test
    print("Camera Masking Demo")
    print("=" * 50)
    
    masker = CameraMasker(
        cameras=["cam_overhead", "cam_ego", "cam_external"],
        mask_prob=0.3,
        min_cameras=1,
    )
    
    # Simulate some masks
    print("\nGenerating 10 random masks (prob=0.3):")
    for i in range(10):
        mask = masker.generate_mask()
        visible = [k for k, v in mask.items() if v]
        masked = [k for k, v in mask.items() if not v]
        print(f"  {i+1}. Visible: {visible}, Masked: {masked}")
    
    # Test with numpy arrays
    print("\nApplying mask to numpy images:")
    fake_images = {
        "cam_overhead": np.random.rand(480, 640, 3).astype(np.float32),
        "cam_ego": np.random.rand(480, 640, 3).astype(np.float32),
        "cam_external": np.random.rand(480, 640, 3).astype(np.float32),
    }
    
    masked, mask = masker.apply(fake_images)
    for cam, visible in mask.items():
        status = "visible" if visible else "MASKED (zeros)"
        mean_val = masked[cam].mean()
        print(f"  {cam}: {status}, mean={mean_val:.4f}")
    
    print("\n" + "=" * 50)
    print("Training integration example:")
    print(TRAINING_EXAMPLE)


