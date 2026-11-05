from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import TASK_CONDITION_NAMES

LOCALIZATION_LABEL_NAMES = ("target_center", "plug_tip")


class _DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = _DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = _DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class TaskConditionedPortLocalizer(nn.Module):
    """Task-conditioned U-Net for target-center / plug-tip localization."""

    def __init__(
        self,
        *,
        image_channels: int = 3,
        condition_dim: int = len(TASK_CONDITION_NAMES),
        base_channels: int = 32,
        point_channels: int = len(LOCALIZATION_LABEL_NAMES),
    ) -> None:
        super().__init__()
        self.image_channels = image_channels
        self.condition_dim = condition_dim
        self.base_channels = base_channels
        self.point_channels = point_channels

        in_channels = image_channels + condition_dim
        self.stem = _DoubleConv(in_channels, base_channels)
        self.down1 = _Down(base_channels, base_channels * 2)
        self.down2 = _Down(base_channels * 2, base_channels * 4)
        self.down3 = _Down(base_channels * 4, base_channels * 8)

        self.up1 = _Up(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = _Up(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up3 = _Up(base_channels * 2, base_channels, base_channels)

        self.heatmap_head = nn.Conv2d(base_channels, point_channels, kernel_size=1)
        self.visibility_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(base_channels * 8, base_channels * 2),
            nn.GELU(),
            nn.Linear(base_channels * 2, point_channels),
        )

    def forward(
        self,
        images: torch.Tensor,
        task_condition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if images.ndim != 4:
            raise ValueError(f"Expected image batch shaped (B, C, H, W), got {images.shape!r}")
        if task_condition.ndim == 1:
            task_condition = task_condition.unsqueeze(0)
        if task_condition.ndim != 2:
            raise ValueError(
                f"Expected task condition shaped (B, D), got {task_condition.shape!r}"
            )

        batch_size, _, height, width = images.shape
        if task_condition.shape[0] != batch_size:
            raise ValueError(
                "Image and condition batch sizes must match: "
                f"{images.shape[0]} vs {task_condition.shape[0]}"
            )

        condition_planes = task_condition.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, height, width
        )
        x = torch.cat([images, condition_planes], dim=1)

        enc1 = self.stem(x)
        enc2 = self.down1(enc1)
        enc3 = self.down2(enc2)
        bottleneck = self.down3(enc3)

        dec2 = self.up1(bottleneck, enc3)
        dec1 = self.up2(dec2, enc2)
        dec0 = self.up3(dec1, enc1)

        return {
            "heatmap_logits": self.heatmap_head(dec0),
            "visibility_logits": self.visibility_head(bottleneck),
        }

    @torch.no_grad()
    def predict_proba(
        self,
        images: torch.Tensor,
        task_condition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = self.forward(images, task_condition)
        return {
            **outputs,
            "heatmap_probs": torch.sigmoid(outputs["heatmap_logits"]),
            "visibility_probs": torch.sigmoid(outputs["visibility_logits"]),
        }

    def model_kwargs(self) -> dict[str, Any]:
        return {
            "image_channels": self.image_channels,
            "condition_dim": self.condition_dim,
            "base_channels": self.base_channels,
            "point_channels": self.point_channels,
        }


def save_port_localizer_checkpoint(
    path: str | Path,
    *,
    model: TaskConditionedPortLocalizer,
    epoch: int,
    step: int,
    optimizer: torch.optim.Optimizer | None = None,
    metrics: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "model_kwargs": model.model_kwargs(),
        "epoch": int(epoch),
        "step": int(step),
        "metrics": metrics or {},
        "config": config or {},
        "task_condition_names": list(TASK_CONDITION_NAMES),
        "label_names": list(LOCALIZATION_LABEL_NAMES),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, Path(path))


def load_port_localizer_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[TaskConditionedPortLocalizer, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    model_kwargs = dict(payload.get("model_kwargs", {}))
    model = TaskConditionedPortLocalizer(**model_kwargs)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload
