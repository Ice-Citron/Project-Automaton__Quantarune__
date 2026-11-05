"""Wrapper for openpi pi0.5 LoRA fine-tuning.

Provides a common interface matching FlowMatchingModel so both can be
used interchangeably in the training loop and inference pipeline.

Requires the openpi package: pip install openpi
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Pi05Wrapper(nn.Module):
    """Wrapper around openpi's pi0.5 for LoRA fine-tuning.

    Adapts the AIC data format to openpi's expected input schema and
    provides the same forward/sample interface as FlowMatchingModel.
    """

    def __init__(self, cfg):
        super().__init__()

        self.action_dim = self._get_action_dim(cfg.action.mode)
        self.horizon = cfg.action.horizon
        self.predict_stiffness = cfg.action.predict_stiffness

        try:
            from openpi.models import load_pi0_5
            from openpi.training import LoRAConfig

            logger.info(f"Loading openpi pi0.5 model: {cfg.openpi_model}")
            self.pi05_model = load_pi0_5(cfg.openpi_model)

            # Configure LoRA
            lora_config = LoRAConfig(
                rank=cfg.lora_rank,
                alpha=cfg.lora_alpha,
            )
            self.pi05_model.apply_lora(lora_config)

            if cfg.freeze_backbone:
                self._freeze_backbone()

            self._openpi_available = True
            logger.info(
                f"Pi0.5 loaded with LoRA rank={cfg.lora_rank}, "
                f"alpha={cfg.lora_alpha}"
            )

        except ImportError:
            logger.warning(
                "openpi not available. Pi05Wrapper will use a placeholder. "
                "Install with: pip install openpi"
            )
            self._openpi_available = False
            self._build_placeholder(cfg)

    @staticmethod
    def _get_action_dim(mode: str) -> int:
        if mode == "velocity":
            return 6
        elif mode == "position":
            return 9
        else:
            raise ValueError(f"Unknown action mode: {mode}")

    def _freeze_backbone(self):
        """Freeze all parameters except LoRA adapters."""
        for name, param in self.pi05_model.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False

    def _build_placeholder(self, cfg):
        """Build a placeholder model matching FlowMatchingModel interface."""
        from quantarune_training.models.flow_matching import FlowMatchingModel

        # Reuse the custom model as a fallback
        self._placeholder = FlowMatchingModel(cfg)
        logger.warning("Using FlowMatchingModel as Pi0.5 placeholder")

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity field (matches FlowMatchingModel interface).

        Args:
            x_t: (B, H, action_dim) noisy action chunk.
            t: (B,) timestep in [0, 1].
            images: Dict of camera tensors, each (B, C, H, W).
            state: (B, state_dim) normalized state.

        Returns:
            (B, H, action_dim) predicted velocity field.
        """
        if not self._openpi_available:
            return self._placeholder.forward(x_t, t, images, state)

        # Convert AIC observations to openpi format
        obs = self._convert_observations(images, state)
        return self.pi05_model.predict_velocity(x_t, t, obs)

    def _convert_observations(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
    ) -> dict:
        """Convert AIC observation format to openpi expected format."""
        obs = {
            "state": state,
        }
        for key, img in images.items():
            obs[key] = img
        return obs

    @torch.no_grad()
    def sample(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Sample action chunk via Euler ODE integration.

        Args:
            images: Dict of camera tensors.
            state: (B, state_dim) normalized state.
            num_steps: Number of ODE integration steps.

        Returns:
            (B, H, action_dim) sampled action chunk.
        """
        if not self._openpi_available:
            return self._placeholder.sample(images, state, num_steps)

        B = state.shape[0]
        device = state.device
        x = torch.randn(B, self.horizon, self.action_dim, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self.forward(x, t, images, state)
            x = x + v * dt

        return x
