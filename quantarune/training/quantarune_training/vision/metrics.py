from __future__ import annotations

import math

import torch


def _binary_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_b = pred > 0.5
    target_b = target > 0.5
    intersection = (pred_b & target_b).sum(dim=(-2, -1)).float()
    union = (pred_b | target_b).sum(dim=(-2, -1)).float()
    both_empty = union == 0
    iou = (intersection + eps) / (union + eps)
    return torch.where(both_empty, torch.ones_like(iou), iou)


def _binary_dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_b = pred > 0.5
    target_b = target > 0.5
    intersection = (pred_b & target_b).sum(dim=(-2, -1)).float()
    denom = pred_b.sum(dim=(-2, -1)).float() + target_b.sum(dim=(-2, -1)).float()
    both_empty = denom == 0
    dice = (2.0 * intersection + eps) / (denom + eps)
    return torch.where(both_empty, torch.ones_like(dice), dice)


def _centroid(mask: torch.Tensor) -> tuple[float, float] | None:
    ys, xs = torch.nonzero(mask > 0.5, as_tuple=True)
    if len(xs) == 0:
        return None
    return float(xs.float().mean().item()), float(ys.float().mean().item())


def _center_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_center = _centroid(pred)
    target_center = _centroid(target)
    if pred_center is None or target_center is None:
        return math.nan
    dx = pred_center[0] - target_center[0]
    dy = pred_center[1] - target_center[1]
    return math.sqrt((dx * dx) + (dy * dy))


@torch.no_grad()
def compute_batch_metrics(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
) -> dict[str, float]:
    mask_probs = torch.sigmoid(outputs["mask_logits"])
    presence_probs = torch.sigmoid(outputs["presence_logits"])
    presence_targets = targets.flatten(start_dim=2).amax(dim=-1)
    presence_pred = (presence_probs > 0.5).float()

    target_iou = _binary_iou(mask_probs[:, 0], targets[:, 0]).mean().item()
    plug_iou = _binary_iou(mask_probs[:, 1], targets[:, 1]).mean().item()
    target_dice = _binary_dice(mask_probs[:, 0], targets[:, 0]).mean().item()
    plug_dice = _binary_dice(mask_probs[:, 1], targets[:, 1]).mean().item()
    presence_acc = (presence_pred == presence_targets).float().mean().item()

    target_center_errors = [
        _center_error(mask_probs[i, 0], targets[i, 0]) for i in range(targets.shape[0])
    ]
    plug_center_errors = [
        _center_error(mask_probs[i, 1], targets[i, 1]) for i in range(targets.shape[0])
    ]

    def _nanmean(values: list[float]) -> float:
        valid = [value for value in values if not math.isnan(value)]
        if not valid:
            return math.nan
        return float(sum(valid) / len(valid))

    return {
        "target_iou": float(target_iou),
        "plug_iou": float(plug_iou),
        "mean_iou": float((target_iou + plug_iou) / 2.0),
        "target_dice": float(target_dice),
        "plug_dice": float(plug_dice),
        "mean_dice": float((target_dice + plug_dice) / 2.0),
        "presence_accuracy": float(presence_acc),
        "target_center_error_px": _nanmean(target_center_errors),
        "plug_center_error_px": _nanmean(plug_center_errors),
    }
