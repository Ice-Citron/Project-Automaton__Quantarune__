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


def _point_from(value: Any) -> np.ndarray | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return np.asarray([float(value[0]), float(value[1])], dtype=np.float32)
    except (TypeError, ValueError):
        return None


def _gaussian_heatmap(
    *,
    width: int,
    height: int,
    point_xy: np.ndarray | None,
    sigma: float,
) -> np.ndarray:
    if point_xy is None:
        return np.zeros((height, width), dtype=np.float32)

    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    dx = xx - float(point_xy[0])
    dy = yy - float(point_xy[1])
    heatmap = np.exp(-((dx * dx) + (dy * dy)) / max(2.0 * sigma * sigma, 1e-6))
    return heatmap.astype(np.float32)


class TaskConditionedLocalizationDataset(Dataset):
    """Dataset for task-conditioned target-center and plug-tip localization."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        root_dir: str | Path | None = None,
        image_size: tuple[int, int] | None = (224, 224),
        augment: bool = False,
        sigma_px: float = 4.0,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.root_dir = (
            Path(root_dir).expanduser().resolve()
            if root_dir is not None
            else self.manifest_path.parent.resolve()
        )
        self.image_size = image_size
        self.augment = augment
        self.sigma_px = float(sigma_px)
        self.records = _load_jsonl(self.manifest_path)

        if not self.records:
            raise ValueError(f"No records found in manifest: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.records[index])
        image = self._read_rgb(self._resolve(record["image_path"]))
        orig_height, orig_width = image.shape[:2]

        target_point = _point_from(record.get("target_center_px"))
        plug_point = _point_from(record.get("plug_tip_px"))
        target_visible = bool(record.get("target_visible", target_point is not None))
        plug_visible = bool(record.get("plug_visible", plug_point is not None))

        if self.image_size is not None:
            width, height = self.image_size
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            height, width = orig_height, orig_width

        sx = float(width) / float(orig_width)
        sy = float(height) / float(orig_height)

        def scale_point(point: np.ndarray | None, visible: bool) -> np.ndarray | None:
            if not visible or point is None:
                return None
            return np.asarray([point[0] * sx, point[1] * sy], dtype=np.float32)

        target_point_scaled = scale_point(target_point, target_visible)
        plug_point_scaled = scale_point(plug_point, plug_visible)

        if self.augment:
            image = self._augment_rgb(image)

        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)).copy()).float()
        image_tensor /= 255.0

        heatmaps = np.stack(
            [
                _gaussian_heatmap(
                    width=width,
                    height=height,
                    point_xy=target_point_scaled,
                    sigma=self.sigma_px,
                ),
                _gaussian_heatmap(
                    width=width,
                    height=height,
                    point_xy=plug_point_scaled,
                    sigma=self.sigma_px,
                ),
            ],
            axis=0,
        )

        points = np.asarray(
            [
                target_point_scaled if target_point_scaled is not None else [-1.0, -1.0],
                plug_point_scaled if plug_point_scaled is not None else [-1.0, -1.0],
            ],
            dtype=np.float32,
        )
        visibility = np.asarray(
            [1.0 if target_point_scaled is not None else 0.0, 1.0 if plug_point_scaled is not None else 0.0],
            dtype=np.float32,
        )

        metadata = {
            "image_path": record["image_path"],
            "episode_id": record.get("episode_id"),
            "frame_idx": record.get("frame_idx"),
            "task_text": record.get("task_text"),
            "target_slot": record.get("target_slot"),
            "port_name": record.get("port_name"),
            "label_source": record.get("label_source"),
            "condition_names": TASK_CONDITION_NAMES,
        }

        return {
            "image": image_tensor,
            "heatmaps": torch.from_numpy(heatmaps),
            "points": torch.from_numpy(points),
            "visibility": torch.from_numpy(visibility),
            "task_condition": torch.from_numpy(task_condition_array_from_mapping(record)),
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
