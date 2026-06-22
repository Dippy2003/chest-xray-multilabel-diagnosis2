"""
ResNet50 transfer learning model.
"""
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNetTransfer(nn.Module):
    """
    ImageNet-pretrained ResNet50 with a new FC head for our 5 classes.

    Freezes everything except layer4 (last residual block) and the FC head.
    Reasoning: only ~3.9k train images, so fine-tuning the whole network risks
    overfitting / wrecking the pretrained low-level filters for no benefit.
    Freezing early layers (edges/textures) and only adapting layer4 (more
    task-specific features) + the head is the standard small-data recipe.
    """

    def __init__(self, num_classes: int = 5, freeze_until_layer4: bool = True):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        if freeze_until_layer4:
            self._freeze_until_layer4()

    def _freeze_until_layer4(self) -> None:
        for name, param in self.backbone.named_parameters():
            if not (name.startswith("layer4") or name.startswith("fc")):
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
