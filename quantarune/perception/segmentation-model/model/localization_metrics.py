from __future__ import annotations

import math

import torch

from .localization_losses import spatial_softargmax2d


def _point_error(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    visibility: torch.Tensor,
    channel_index: int,
) -> list[float]:
    errors: list[float] = []
    for batch_idx in range(pred_points.shape[0]):
        if float(visibility[batch_idx, channel_index].item()) <= 0.5:
            continue
        dx = float(pred_points[batch_idx, channel_index, 0].item() - target_points[batch_idx, channel_index, 0].item())
        dy = float(pred_points[batch_idx, channel_index, 1].item() - target_points[batch_idx, channel_index, 1].item())
        errors.append(math.sqrt((dx * dx) + (dy * dy)))
    return errors


def _nanmean(values: list[float]) -> float:
    if not values:
        return math.nan
    return float(sum(values) / len(values))


@torch.no_grad()
def compute_localization_metrics(
    outputs: dict[str, torch.Tensor],
    target_points: torch.Tensor,
    target_visibility: torch.Tensor,
) -> dict[str, float]:
    pred_points = spatial_softargmax2d(outputs["heatmap_logits"])
    visibility_probs = torch.sigmoid(outputs["visibility_logits"])
    visibility_pred = (visibility_probs > 0.5).float()

    target_errors = _point_error(pred_points, target_points, target_visibility, channel_index=0)
    plug_errors = _point_error(pred_points, target_points, target_visibility, channel_index=1)

    target_vis_acc = (
        visibility_pred[:, 0] == target_visibility[:, 0]
    ).float().mean().item()
    plug_vis_acc = (
        visibility_pred[:, 1] == target_visibility[:, 1]
    ).float().mean().item()

    mean_point_error = _nanmean(target_errors + plug_errors)

    return {
        "target_center_error_px": _nanmean(target_errors),
        "plug_tip_error_px": _nanmean(plug_errors),
        "mean_point_error_px": mean_point_error,
        "target_visibility_accuracy": float(target_vis_acc),
        "plug_visibility_accuracy": float(plug_vis_acc),
        "mean_visibility_accuracy": float((target_vis_acc + plug_vis_acc) / 2.0),
    }
