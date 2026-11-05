from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs_flat = probs.flatten(start_dim=2)
    targets_flat = targets.flatten(start_dim=2)
    intersection = (probs_flat * targets_flat).sum(dim=-1)
    denom = probs_flat.sum(dim=-1) + targets_flat.sum(dim=-1)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def segmentation_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    *,
    presence_weight: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask_logits = outputs["mask_logits"]
    presence_logits = outputs["presence_logits"]

    bce = F.binary_cross_entropy_with_logits(mask_logits, targets)
    dice = soft_dice_loss(mask_logits, targets)

    presence_targets = targets.flatten(start_dim=2).amax(dim=-1)
    presence_bce = F.binary_cross_entropy_with_logits(
        presence_logits,
        presence_targets,
    )

    total = bce + dice + (presence_weight * presence_bce)
    return total, {
        "loss_total": total.detach(),
        "loss_bce": bce.detach(),
        "loss_dice": dice.detach(),
        "loss_presence": presence_bce.detach(),
    }
