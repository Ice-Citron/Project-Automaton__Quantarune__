"""Normalization statistics computation and application.

Computes per-feature mean/std for states, actions, and per-camera image
statistics, stored as safetensors matching LeRobot convention.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file, load_file


def compute_stats(
    states: np.ndarray,
    action_chunks: np.ndarray,
    images: dict[str, list[np.ndarray]] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute normalization statistics from training data.

    Args:
        states: (N, state_dim) state vectors.
        action_chunks: (N, H, action_dim) action chunks.
        images: Optional dict mapping camera name to list of images (H,W,C uint8).

    Returns:
        Dict of statistics tensors keyed by feature name.
    """
    stats = {}

    # State stats
    states_t = torch.from_numpy(states).float()
    stats["state.mean"] = states_t.mean(dim=0)
    stats["state.std"] = states_t.std(dim=0).clamp(min=1e-6)

    # Action stats (flatten chunks for per-dim stats)
    actions_flat = torch.from_numpy(
        action_chunks.reshape(-1, action_chunks.shape[-1])
    ).float()
    stats["action.mean"] = actions_flat.mean(dim=0)
    stats["action.std"] = actions_flat.std(dim=0).clamp(min=1e-6)

    # Image stats
    if images is not None:
        for cam_name, cam_images in images.items():
            # Sample subset for efficiency
            sample_size = min(len(cam_images), 1000)
            indices = np.random.choice(len(cam_images), sample_size, replace=False)
            sampled = np.stack([cam_images[i] for i in indices]).astype(np.float32) / 255.0
            # (N, H, W, C) -> (N, C, H, W)
            sampled = np.transpose(sampled, (0, 3, 1, 2))
            sampled_t = torch.from_numpy(sampled)
            stats[f"image.{cam_name}.mean"] = sampled_t.mean(dim=(0, 2, 3))
            stats[f"image.{cam_name}.std"] = sampled_t.std(dim=(0, 2, 3)).clamp(min=1e-6)

    return stats


def save_stats(stats: dict[str, torch.Tensor], path: Path) -> None:
    """Save normalization statistics as safetensors."""
    save_file(stats, str(path))


def load_stats(path: Path) -> dict[str, torch.Tensor]:
    """Load normalization statistics from safetensors."""
    return load_file(str(path))


class Normalizer:
    """Apply normalization to states, actions, and images."""

    def __init__(self, stats: dict[str, torch.Tensor], device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.stats = {k: v.to(self.device) for k, v in stats.items()}

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.stats["state.mean"]) / self.stats["state.std"]

    def unnormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return state * self.stats["state.std"] + self.stats["state.mean"]

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.stats["action.mean"]) / self.stats["action.std"]

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.stats["action.std"] + self.stats["action.mean"]

    def normalize_image(self, image: torch.Tensor, camera: str) -> torch.Tensor:
        """Normalize image tensor (B, C, H, W) or (C, H, W)."""
        mean = self.stats[f"image.{camera}.mean"]
        std = self.stats[f"image.{camera}.std"]
        if image.dim() == 4:
            mean = mean.view(1, 3, 1, 1)
            std = std.view(1, 3, 1, 1)
        else:
            mean = mean.view(3, 1, 1)
            std = std.view(3, 1, 1)
        return (image - mean) / std
