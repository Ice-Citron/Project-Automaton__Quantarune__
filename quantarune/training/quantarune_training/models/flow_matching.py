"""Flow matching denoiser using DiT (Diffusion Transformer) architecture.

Learns a velocity field v(x_t, t, c) that transports noise to the action
distribution conditioned on observations. Uses adaptive layer normalization
(AdaLN) for timestep conditioning and cross-attention for observation conditioning.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from quantarune_training.models.image_encoder import ImageEncoder
from quantarune_training.models.state_encoder import StateEncoder


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) timestep in [0, 1].
        Returns:
            (B, dim) embedding.
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization conditioned on timestep embedding."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(cond_dim, dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale_shift = self.linear(cond)
        if scale_shift.dim() == 2:
            scale_shift = scale_shift.unsqueeze(1)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class DiTBlock(nn.Module):
    """Transformer block with AdaLN, self-attention, cross-attention, and MLP."""

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        # Self-attention with AdaLN
        self.adanorm1 = AdaLayerNorm(hidden_dim, cond_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention to observation conditioning
        self.adanorm2 = AdaLayerNorm(hidden_dim, cond_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # MLP with AdaLN
        self.adanorm3 = AdaLayerNorm(hidden_dim, cond_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        obs_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) action tokens.
            cond: (B, cond_dim) timestep + obs conditioning.
            obs_tokens: (B, N, D) observation tokens for cross-attention.

        Returns:
            (B, T, D) updated tokens.
        """
        # Self-attention
        h = self.adanorm1(x, cond)
        h, _ = self.self_attn(h, h, h)
        x = x + h

        # Cross-attention
        h = self.adanorm2(x, cond)
        h, _ = self.cross_attn(h, obs_tokens, obs_tokens)
        x = x + h

        # MLP
        h = self.adanorm3(x, cond)
        x = x + self.mlp(h)

        return x


class FlowMatchingModel(nn.Module):
    """Flow matching model with DiT denoiser.

    Predicts the velocity field v(x_t, t, obs) for flow matching,
    where x_t is the noisy action chunk at timestep t, conditioned
    on observations (images + robot state).
    """

    def __init__(self, cfg):
        super().__init__()

        self.action_dim = self._get_action_dim(cfg.action.mode)
        self.horizon = cfg.action.horizon
        hidden_dim = cfg.denoiser.hidden_dim
        cond_dim = hidden_dim  # Conditioning and hidden share dim

        # Observation encoders
        self.image_encoder = ImageEncoder(
            backbone=cfg.image_encoder.backbone,
            pretrained=cfg.image_encoder.pretrained,
            freeze_until_block=cfg.image_encoder.freeze_until_block,
            feature_dim=cfg.image_encoder.feature_dim,
        )
        self.state_encoder = StateEncoder(
            input_dim=cfg.state_encoder.input_dim,
            hidden_dims=list(cfg.state_encoder.hidden_dims),
            output_dim=cfg.state_encoder.output_dim,
        )

        # Combine image + state features
        self.obs_projection = nn.Sequential(
            nn.Linear(cfg.image_encoder.feature_dim + cfg.state_encoder.output_dim, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.GELU(),
        )

        # Observation tokens for cross-attention (project to sequence of tokens)
        self.obs_to_tokens = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 4),
            nn.GELU(),
        )

        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Action input projection
        self.action_input_proj = nn.Linear(self.action_dim, hidden_dim)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=hidden_dim,
                cond_dim=cond_dim,
                num_heads=cfg.denoiser.num_heads,
                dropout=cfg.denoiser.dropout,
            )
            for _ in range(cfg.denoiser.num_blocks)
        ])

        # Output projection
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.action_dim)

        # Optional stiffness head
        self.predict_stiffness = cfg.action.predict_stiffness
        if self.predict_stiffness:
            self.stiffness_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, cfg.action.stiffness_dim),
                nn.Softplus(),  # Stiffness must be positive
            )

    @staticmethod
    def _get_action_dim(mode: str) -> int:
        if mode == "velocity":
            return 6
        elif mode == "position":
            return 9
        else:
            raise ValueError(f"Unknown action mode: {mode}")

    def encode_observations(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode observations into conditioning vector and tokens.

        Returns:
            cond: (B, cond_dim) conditioning vector.
            obs_tokens: (B, 4, cond_dim) observation tokens for cross-attention.
        """
        img_feat = self.image_encoder(images)
        state_feat = self.state_encoder(state)
        obs_feat = self.obs_projection(torch.cat([img_feat, state_feat], dim=-1))

        # Create observation tokens for cross-attention
        obs_tokens = self.obs_to_tokens(obs_feat)  # (B, cond_dim * 4)
        cond_dim = obs_feat.shape[-1]
        obs_tokens = obs_tokens.reshape(-1, 4, cond_dim)  # (B, 4, cond_dim)

        return obs_feat, obs_tokens

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity field.

        Args:
            x_t: (B, H, action_dim) noisy action chunk.
            t: (B,) timestep in [0, 1].
            images: Dict of camera tensors, each (B, C, H, W).
            state: (B, state_dim) normalized state.

        Returns:
            (B, H, action_dim) predicted velocity field.
        """
        B = x_t.shape[0]

        # Encode observations
        obs_cond, obs_tokens = self.encode_observations(images, state)

        # Timestep embedding
        t_emb = self.time_embed(t)  # (B, cond_dim)

        # Combined conditioning
        cond = obs_cond + t_emb  # (B, cond_dim)

        # Project action tokens
        tokens = self.action_input_proj(x_t)  # (B, H, hidden_dim)

        # DiT blocks
        for block in self.blocks:
            tokens = block(tokens, cond, obs_tokens)

        # Output projection
        tokens = self.final_norm(tokens)
        velocity = self.output_proj(tokens)  # (B, H, action_dim)

        return velocity

    def predict_stiffness_values(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
    ) -> torch.Tensor | None:
        """Predict stiffness values from observations (not flow-matched).

        Args:
            images: Dict of camera tensors, each (B, C, H, W).
            state: (B, state_dim) normalized state.

        Returns:
            (B, stiffness_dim) predicted stiffness, or None if disabled.
        """
        if not self.predict_stiffness:
            return None

        obs_cond, _ = self.encode_observations(images, state)
        return self.stiffness_head(obs_cond)

    @torch.no_grad()
    def sample(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Sample action chunk via Euler ODE integration.

        Args:
            images: Dict of camera tensors, each (B, C, H, W).
            state: (B, state_dim) normalized state.
            num_steps: Number of ODE integration steps.

        Returns:
            (B, H, action_dim) sampled action chunk.
        """
        B = state.shape[0]
        device = state.device

        # Start from noise
        x = torch.randn(B, self.horizon, self.action_dim, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self.forward(x, t, images, state)
            x = x + v * dt

        return x
