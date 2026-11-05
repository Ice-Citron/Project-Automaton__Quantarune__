"""Flow matching loss and optional stiffness loss."""

import torch
import torch.nn.functional as F


def flow_matching_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Compute flow matching loss.

    Flow matching interpolates between noise x_0 ~ N(0,I) and target x_1,
    with x_t = (1-t)*x_0 + t*x_1. The model predicts the velocity field
    v = x_1 - x_0, and the loss is MSE between predicted and target velocity.

    Args:
        model: FlowMatchingModel or Pi05Wrapper.
        batch: Dict with keys: state, action_chunk, images.*.
        device: Torch device.

    Returns:
        Dict with loss components: flow_loss, per_dim_mse, and optionally
        stiffness_loss.
    """
    action_chunk = batch["action_chunk"].to(device)  # (B, H, action_dim)
    state = batch["state"].to(device)
    B = action_chunk.shape[0]

    # Collect images
    images = {}
    for key in batch:
        if key.startswith("images."):
            images[key] = batch[key].to(device)

    # Sample timestep and noise
    t = torch.rand(B, device=device)
    noise = torch.randn_like(action_chunk)

    # Interpolate: x_t = (1 - t) * noise + t * action
    t_expanded = t[:, None, None]  # (B, 1, 1)
    x_t = (1.0 - t_expanded) * noise + t_expanded * action_chunk

    # Target velocity
    target_v = action_chunk - noise

    # Predict velocity
    pred_v = model(x_t, t, images, state)

    # Flow matching loss
    flow_loss = F.mse_loss(pred_v, target_v)

    # Per-dimension MSE (for logging, detached)
    with torch.no_grad():
        per_dim_mse = ((pred_v - target_v) ** 2).mean(dim=(0, 1))  # (action_dim,)

    losses = {
        "flow_loss": flow_loss,
        "per_dim_mse": per_dim_mse,
    }

    # Optional stiffness loss
    if hasattr(model, "predict_stiffness") and model.predict_stiffness:
        if "stiffness_chunk" in batch:
            stiffness_gt = batch["stiffness_chunk"].to(device)
            stiffness_pred = model.predict_stiffness_values(images, state)
            if stiffness_pred is not None:
                losses["stiffness_loss"] = F.mse_loss(stiffness_pred, stiffness_gt)

    return losses


@torch.no_grad()
def compute_action_mse(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    num_ode_steps: int = 10,
) -> torch.Tensor:
    """Compute MSE between sampled actions and ground truth.

    This is a more meaningful metric than flow loss, as it measures
    the actual action prediction quality after ODE integration.

    Args:
        model: FlowMatchingModel or Pi05Wrapper.
        batch: Dict with keys: state, action_chunk, images.*.
        device: Torch device.
        num_ode_steps: Number of ODE integration steps.

    Returns:
        Scalar MSE tensor.
    """
    action_chunk = batch["action_chunk"].to(device)
    state = batch["state"].to(device)

    images = {}
    for key in batch:
        if key.startswith("images."):
            images[key] = batch[key].to(device)

    # Sample actions via ODE
    pred_actions = model.sample(images, state, num_steps=num_ode_steps)

    return F.mse_loss(pred_actions, action_chunk)
