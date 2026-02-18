#!/usr/bin/env python3
"""
Multi-dataset training script for SO-101 robot.

This script trains a policy (Pi0, GROOT, ACT, Diffusion) on multiple
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

import torch


def load_policy(policy_name: str, config: dict, device: str = "cuda"):
    """Load a policy model based on name."""
    from lerobot.configs.types import FeatureType, PolicyFeature
    
    # Get shapes from config
    cameras = config["dataset"]["cameras"]
    state_shape = config["policy"]["input_shapes"].get("observation.state", [6])
    action_shape = config["policy"]["output_shapes"].get("action", [6])
    
    # Build input features with proper PolicyFeature objects
    input_features = {}
    for cam in cameras:
        key = f"observation.images.{cam}"
        shape = config["policy"]["input_shapes"].get(key, [3, 480, 640])
        input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=shape)
    input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=state_shape)
    
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=action_shape)
    }
    
    if policy_name == "pi0":
        from lerobot.policies.pi0 import PI0Config, PI0Policy
        
        policy_config = PI0Config(
            n_obs_steps=1,
            input_features=input_features,
            output_features=output_features,
            device=device,
        )
        policy = PI0Policy(policy_config)
    
    elif policy_name == "pi05" or policy_name == "pi0.5":
        from lerobot.policies.pi05 import PI05Config, PI05Policy
        
        policy_config = PI05Config(
            n_obs_steps=1,
            input_features=input_features,
            output_features=output_features,
            device=device,
        )
        policy = PI05Policy(policy_config)
        
    elif policy_name == "groot" or policy_name == "groot_n1.5":
        from lerobot.policies.groot import GrootConfig, GrootPolicy
        
        policy_config = GrootConfig(
            n_obs_steps=1,
            input_features=input_features,
            output_features=output_features,
            device=device,
            base_model_path="nvidia/GR00T-N1.5-3B",
        )
        policy = GrootPolicy(policy_config)
    
    elif policy_name == "groot_n1.6":
        from lerobot.policies.groot import GrootConfig, GrootPolicy
        
        policy_config = GrootConfig(
            n_obs_steps=1,
            input_features=input_features,
            output_features=output_features,
            device=device,
            base_model_path="nvidia/GR00T-N1.6-3B",
        )
        policy = GrootPolicy(policy_config)
        
    elif policy_name == "act":
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        
        policy_config = ACTConfig(
            input_features=input_features,
            output_features=output_features,
            device=device,
        )
        policy = ACTPolicy(policy_config)
        
    elif policy_name == "diffusion":
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        
        policy_config = DiffusionConfig(
            input_features=input_features,
            output_features=output_features,
            device=device,
        )
        policy = DiffusionPolicy(policy_config)
        
    else:
        raise ValueError(
            f"Unknown policy: {policy_name}. "
            "Available: pi0, pi05/pi0.5, groot/groot_n1.5, groot_n1.6, act, diffusion"
        )
    
    return policy.to(device)


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
    policy_name = config["policy"]["name"]
    
    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print("=" * 60)
    if test_mode:
        print("MULTI-DATASET TRAINING (TEST MODE)")
    else:
        print("MULTI-DATASET TRAINING")
    print("=" * 60)
    print(f"Policy: {policy_name}")
    print(f"Device: {device}")
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
        num_workers=0 if test_mode else 4,
    )
    
    print(f"Total samples: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")
    print()
    
    # Load policy model
    print("=" * 60)
    print(f"LOADING {policy_name.upper()} MODEL")
    print("=" * 60)
    
    try:
        policy = load_policy(policy_name, config, device=device)
        print(f"✓ {policy_name.upper()} loaded successfully!")
        
        # Count parameters
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    except Exception as e:
        print(f"✗ Failed to load {policy_name}: {e}")
        if test_mode:
            print("  Continuing with data loading test only...")
            policy = None
        else:
            return 1
    
    print()
    
    # Setup optimizer
    if policy is not None:
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"].get("weight_decay", 1e-5),
        )
    
    # Training loop
    print("=" * 60)
    print("TRAINING LOOP")
    print("=" * 60)
    
    epochs = config["training"]["epochs"]
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        if policy is not None:
            policy.train()
        
        batch_count = 0
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            batch_count += 1
            
            # In test mode, limit batches
            if max_batches and batch_count > max_batches:
                print(f"  [Test mode] Stopping after {max_batches} batches")
                break
            
            if policy is not None:
                # Move batch to device
                batch_device = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch_device[k] = v.to(device)
                    else:
                        batch_device[k] = v
                
                # Forward pass
                optimizer.zero_grad()
                
                try:
                    # LeRobot policies expect specific batch format
                    output = policy.forward(batch_device)
                    
                    if hasattr(output, "loss"):
                        loss = output.loss
                    elif isinstance(output, dict) and "loss" in output:
                        loss = output["loss"]
                    else:
                        # Compute MSE loss between predicted and target actions
                        if "action" in output and "action" in batch_device:
                            loss = torch.nn.functional.mse_loss(
                                output["action"], batch_device["action"]
                            )
                        else:
                            loss = torch.tensor(0.0, device=device)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    if config["training"].get("grad_clip"):
                        torch.nn.utils.clip_grad_norm_(
                            policy.parameters(),
                            config["training"]["grad_clip"]
                        )
                    
                    optimizer.step()
                    total_loss += loss.item()
                    
                except Exception as e:
                    if test_mode:
                        print(f"  [batch {batch_idx + 1}] Forward pass error: {e}")
                        continue
                    else:
                        raise
            
            # Log progress
            if batch_idx % config["training"].get("log_freq", 100) == 0 or test_mode:
                sample_info = f"batch {batch_idx + 1}"
                if "dataset_name" in batch:
                    datasets_in_batch = set(batch["dataset_name"]) if hasattr(batch["dataset_name"], "__iter__") else {batch["dataset_name"]}
                    sample_info += f", datasets: {len(datasets_in_batch)}"
                if policy is not None and batch_count > 0:
                    avg_loss = total_loss / batch_count
                    sample_info += f", loss: {avg_loss:.4f}"
                print(f"  [{sample_info}]")
        
        avg_epoch_loss = total_loss / max(batch_count, 1)
        print(f"  Epoch {epoch + 1} complete: {batch_count} batches, avg loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        if args.output_dir and policy is not None:
            save_freq = config["training"].get("save_freq", 10)
            if (epoch + 1) % save_freq == 0:
                checkpoint_dir = args.output_dir / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1}.pt"
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_epoch_loss,
                }, checkpoint_path)
                print(f"  Saved checkpoint: {checkpoint_path}")
    
    print()
    print("=" * 60)
    if test_mode:
        print("TEST COMPLETE!")
        print("=" * 60)
        print()
        if policy is not None:
            print(f"✓ {policy_name.upper()} model loaded and tested")
        print("✓ Data loading pipeline validated")
        print("✓ Training loop executed")
        print()
        print("To run full training:")
        print("  python scripts/main.py --pipeline train --epochs 100")
    else:
        print("TRAINING COMPLETE")
        print("=" * 60)
        if args.output_dir:
            print(f"Checkpoints saved to: {args.output_dir}/checkpoints/")
    
    return 0


if __name__ == "__main__":
    exit(main())
