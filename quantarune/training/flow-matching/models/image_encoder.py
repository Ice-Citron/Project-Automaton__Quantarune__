"""Image encoder for 3 wrist cameras using pretrained ResNet-18."""

import torch
import torch.nn as nn
import timm


class ImageEncoder(nn.Module):
    """Encode 3 camera images into a single conditioning vector.

    Uses pretrained ResNet-18 backbones (optionally with frozen early layers)
    for each camera, concatenates features, and projects to conditioning dim.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_until_block: int = 2,
        feature_dim: int = 512,
        num_cameras: int = 3,
    ):
        super().__init__()
        self.num_cameras = num_cameras

        # Create backbone
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0
        )
        backbone_dim = self.backbone.num_features

        # Freeze early layers
        if freeze_until_block > 0:
            self._freeze_early_blocks(freeze_until_block)

        # Project concatenated features to conditioning dim
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim * num_cameras, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    def _freeze_early_blocks(self, until_block: int):
        """Freeze conv1, bn1, and the first N residual blocks."""
        # Freeze initial conv and norm
        for param in self.backbone.conv1.parameters():
            param.requires_grad = False
        for param in self.backbone.bn1.parameters():
            param.requires_grad = False

        # Freeze residual blocks
        block_names = ["layer1", "layer2", "layer3", "layer4"]
        for block_name in block_names[:until_block]:
            block = getattr(self.backbone, block_name, None)
            if block is not None:
                for param in block.parameters():
                    param.requires_grad = False

    def forward(self, images: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode camera images.

        Args:
            images: Dict with keys like "images.left_camera", etc.
                Each value is (B, C, H, W) float tensor.

        Returns:
            (B, feature_dim) conditioning vector.
        """
        features = []
        for key in sorted(images.keys()):
            feat = self.backbone(images[key])  # (B, backbone_dim)
            features.append(feat)

        # Concatenate camera features
        concat = torch.cat(features, dim=-1)  # (B, backbone_dim * num_cameras)
        return self.projection(concat)  # (B, feature_dim)
