"""PyTorch Dataset for flow matching training.

Loads converted episodes and returns observation-action pairs with
optional image augmentation.
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from quantarune_training.data.normalization import Normalizer, load_stats

logger = logging.getLogger(__name__)

CAMERA_NAMES = ["left_camera", "center_camera", "right_camera"]


class FlowMatchingDataset(Dataset):
    """Dataset for flow matching policy training.

    Each sample contains:
        - images: dict of camera tensors (C, H, W) float32 [0, 1]
        - state: (state_dim,) float32 normalized
        - action_chunk: (H, action_dim) float32 normalized
    """

    def __init__(
        self,
        data_dir: str,
        normalize: bool = True,
        image_augmentation: bool = False,
    ):
        """
        Args:
            data_dir: Path to converted dataset directory.
            normalize: Whether to apply normalization.
            image_augmentation: Whether to apply random image augmentation.
        """
        self.data_dir = Path(data_dir)
        self.image_augmentation = image_augmentation

        # Load metadata
        with open(self.data_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        self.action_mode = self.metadata["action_mode"]
        self.action_dim = self.metadata["action_dim"]
        self.chunk_horizon = self.metadata["chunk_horizon"]

        # Load normalizer
        self.normalizer = None
        if normalize:
            stats_path = self.data_dir / "normalization_stats.safetensors"
            if stats_path.exists():
                stats = load_stats(stats_path)
                self.normalizer = Normalizer(stats)
            else:
                logger.warning("No normalization stats found, skipping normalization")

        # Load all episodes into memory
        self._load_episodes()

    def _load_episodes(self):
        """Load all episode data into memory."""
        self.states = []
        self.action_chunks = []
        self.images = {cam: [] for cam in CAMERA_NAMES}
        self.has_images = False

        episode_dirs = sorted(self.data_dir.glob("episode_*"))
        for ep_dir in episode_dirs:
            data = torch.load(ep_dir / "data.pt", weights_only=True)
            self.states.append(data["states"])
            self.action_chunks.append(data["action_chunks"])

            # Load images if available
            for cam in CAMERA_NAMES:
                img_path = ep_dir / f"{cam}.npy"
                if img_path.exists():
                    imgs = np.load(img_path, mmap_mode="r")
                    self.images[cam].append(imgs)
                    self.has_images = True

        if not self.states:
            raise ValueError(f"No episodes found in {self.data_dir}")

        # Build index mapping (global_idx -> (episode_idx, local_idx))
        self.index_map = []
        for ep_idx, states in enumerate(self.states):
            for local_idx in range(len(states)):
                self.index_map.append((ep_idx, local_idx))

        logger.info(
            f"Loaded {len(episode_dirs)} episodes, {len(self.index_map)} samples"
        )

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx, local_idx = self.index_map[idx]

        state = self.states[ep_idx][local_idx].float()
        action_chunk = self.action_chunks[ep_idx][local_idx].float()

        # Normalize
        if self.normalizer is not None:
            state = self.normalizer.normalize_state(state)
            action_chunk = self.normalizer.normalize_action(action_chunk)

        sample = {
            "state": state,
            "action_chunk": action_chunk,
        }

        # Load and process images
        if self.has_images:
            for cam in CAMERA_NAMES:
                if self.images[cam] and ep_idx < len(self.images[cam]):
                    img = self.images[cam][ep_idx][local_idx]
                    img = self._process_image(img, cam)
                    sample[f"images.{cam}"] = img

        return sample

    def _process_image(self, img: np.ndarray, camera: str) -> torch.Tensor:
        """Process image: convert to tensor, optionally augment and normalize.

        Args:
            img: (H, W, C) or (C, H, W) uint8 image.
            camera: Camera name for normalization stats.

        Returns:
            (C, H, W) float32 tensor.
        """
        if img.ndim == 3 and img.shape[0] == 3:
            # Already (C, H, W)
            img_chw = img
        else:
            # (H, W, C) -> (C, H, W)
            img_chw = np.transpose(img, (2, 0, 1))

        tensor = torch.from_numpy(img_chw.copy()).float() / 255.0

        if self.image_augmentation:
            tensor = self._augment_image(tensor)

        if self.normalizer is not None:
            tensor = self.normalizer.normalize_image(tensor, camera)

        return tensor

    def _augment_image(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to image tensor (C, H, W)."""
        # Random brightness
        if torch.rand(1).item() < 0.5:
            factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
            img = (img * factor).clamp(0, 1)

        # Random contrast
        if torch.rand(1).item() < 0.5:
            mean = img.mean()
            factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
            img = ((img - mean) * factor + mean).clamp(0, 1)

        return img
