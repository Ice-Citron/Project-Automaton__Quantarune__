from __future__ import annotations

import torch
import torch.nn.functional as F


def spatial_softargmax2d(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError(f"Expected logits shaped (B, K, H, W), got {logits.shape!r}")
    batch_size, channels, height, width = logits.shape
    flat = logits.reshape(batch_size, channels, height * width)
    probs = torch.softmax(flat, dim=-1)

    xs = torch.linspace(0.0, float(width - 1), width, device=logits.device, dtype=logits.dtype)
    ys = torch.linspace(0.0, float(height - 1), height, device=logits.device, dtype=logits.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xx = xx.reshape(1, 1, height * width)
    yy = yy.reshape(1, 1, height * width)

    x = (probs * xx).sum(dim=-1)
    y = (probs * yy).sum(dim=-1)
    return torch.stack([x, y], dim=-1)


def localization_loss(
    outputs: dict[str, torch.Tensor],
    target_heatmaps: torch.Tensor,
    target_points: torch.Tensor,
    target_visibility: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    heatmap_logits = outputs["heatmap_logits"]
    visibility_logits = outputs["visibility_logits"]

    heatmap_probs = torch.sigmoid(heatmap_logits)
    heatmap_loss = F.mse_loss(heatmap_probs, target_heatmaps)

    visibility_loss = F.binary_cross_entropy_with_logits(
        visibility_logits,
        target_visibility,
    )

    pred_points = spatial_softargmax2d(heatmap_logits)
    visible_mask = target_visibility > 0.5

    if bool(visible_mask.any().item()):
        pred_points_visible = pred_points[visible_mask]
        target_points_visible = target_points[visible_mask]
        coord_loss = F.smooth_l1_loss(pred_points_visible, target_points_visible)
    else:
        coord_loss = heatmap_logits.new_zeros(())

    total_loss = heatmap_loss + (0.5 * visibility_loss) + (2.0 * coord_loss)
    return total_loss, {
        "total_loss": total_loss.detach(),
        "heatmap_loss": heatmap_loss.detach(),
        "visibility_loss": visibility_loss.detach(),
        "coord_loss": coord_loss.detach(),
    }
