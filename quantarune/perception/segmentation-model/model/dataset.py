from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import TASK_CONDITION_NAMES, task_condition_array_from_mapping


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class TaskConditionedSegmentationDataset(Dataset):
    """Dataset for task-conditioned target-port and plug segmentation."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        root_dir: str | Path | None = None,
        image_size: tuple[int, int] | None = (224, 224),
        augment: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.root_dir = (
            Path(root_dir).expanduser().resolve()
            if root_dir is not None
            else self.manifest_path.parent.resolve()
        )
        self.image_size = image_size
        self.augment = augment
        self.records = _load_jsonl(self.manifest_path)

        if not self.records:
            raise ValueError(f"No records found in manifest: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.records[index])
        image = self._read_rgb(self._resolve(record["image_path"]))
        target_mask = self._read_mask(self._resolve(record["target_mask_path"]))

        plug_mask_path = record.get("plug_mask_path")
        if isinstance(plug_mask_path, str) and plug_mask_path:
            plug_mask = self._read_mask(self._resolve(plug_mask_path))
        else:
            plug_mask = np.zeros_like(target_mask, dtype=np.uint8)

        if self.image_size is not None:
            width, height = self.image_size
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            target_mask = cv2.resize(
                target_mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            plug_mask = cv2.resize(
                plug_mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

        if self.augment:
            image = self._augment_rgb(image)

        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)).copy()).float()
        image_tensor /= 255.0

        masks = np.stack(
            [
                (target_mask > 127).astype(np.float32),
                (plug_mask > 127).astype(np.float32),
            ],
            axis=0,
        )
        mask_tensor = torch.from_numpy(masks)
        task_condition = torch.from_numpy(task_condition_array_from_mapping(record))

        metadata = {
            "image_path": record["image_path"],
            "episode_id": record.get("episode_id"),
            "frame_idx": record.get("frame_idx"),
            "task_text": record.get("task_text"),
            "target_slot": record.get("target_slot"),
            "port_name": record.get("port_name"),
            "condition_names": TASK_CONDITION_NAMES,
        }

        return {
            "image": image_tensor,
            "masks": mask_tensor,
            "task_condition": task_condition,
            "metadata": metadata,
        }

    def _resolve(self, rel_or_abs: str) -> Path:
        path = Path(rel_or_abs)
        if path.is_absolute():
            return path
        return (self.root_dir / path).resolve()

    @staticmethod
    def _read_rgb(path: Path) -> np.ndarray:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read RGB image: {path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _read_mask(path: Path) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask image: {path}")
        return mask

    @staticmethod
    def _augment_rgb(image: np.ndarray) -> np.ndarray:
        image_f = image.astype(np.float32)

        brightness = np.random.uniform(0.90, 1.10)
        contrast = np.random.uniform(0.90, 1.10)
        image_f = (image_f - 127.5) * contrast + 127.5
        image_f *= brightness

        noise_std = np.random.uniform(0.0, 3.0)
        if noise_std > 0.0:
            image_f += np.random.normal(0.0, noise_std, size=image_f.shape)

        return np.clip(image_f, 0.0, 255.0).astype(np.uint8)
