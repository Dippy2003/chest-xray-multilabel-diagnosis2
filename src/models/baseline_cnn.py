"""
Custom CNN trained from scratch — baseline for comparison against transfer learning.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU -> MaxPool."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineCNN(nn.Module):
    """
    5-block CNN for multi-label chest X-ray classification.

    Uses global average pooling instead of flattening the final feature map —
    this keeps the classifier head small (a flatten at 224x224 input would need
    a ~3M-parameter first FC layer) and makes the network agnostic to input
    resolution, which matters for Grad-CAM visualizations later.

    Outputs raw logits (no sigmoid) — pair with BCEWithLogitsLoss, not BCELoss,
    since sigmoid-then-BCE is numerically unstable for confident wrong
    predictions and most multi-label losses expect logits directly.
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),    # 224 -> 112
            ConvBlock(32, 64),   # 112 -> 56
            ConvBlock(64, 128),  # 56 -> 28
            ConvBlock(128, 256),  # 28 -> 14
            ConvBlock(256, 256),  # 14 -> 7
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)
