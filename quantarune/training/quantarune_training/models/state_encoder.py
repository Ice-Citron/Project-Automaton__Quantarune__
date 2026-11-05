"""MLP encoder for 26D robot state vector."""

import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    """Encode robot state into a conditioning vector.

    Simple MLP with LayerNorm and GELU activations.
    """

    def __init__(
        self,
        input_dim: int = 26,
        hidden_dims: list[int] | None = None,
        output_dim: int = 512,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, input_dim) normalized state vector.

        Returns:
            (B, output_dim) state encoding.
        """
        return self.net(state)
