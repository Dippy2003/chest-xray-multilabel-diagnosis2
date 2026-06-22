"""
EfficientNet-B0 transfer learning model.
"""
import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class EfficientNetTransfer(nn.Module):
    """
    ImageNet-pretrained EfficientNet-B0 with a new classifier head for our 5 classes.

    Same small-data reasoning as ResNetTransfer: freeze most of the backbone,
    only unfreeze the last two feature blocks (features[7], features[8] - the
    final MBConv stage + the 1x1 conv to 1280 channels) plus the head.
    """

    def __init__(self, num_classes: int = 5, freeze_early_layers: bool = True):
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

        if freeze_early_layers:
            self._freeze_early_layers()

    def _freeze_early_layers(self) -> None:
        for param in self.backbone.features[:7].parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
