"""
Grad-CAM for the ResNet50 transfer model - shows which pixels the model
actually looked at for a given predicted class, instead of trusting the
AUC/F1 numbers blindly.
"""
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class GradCAM:
    """
    Hooks a target conv layer, runs a forward + backward pass for one class,
    and turns the gradient-weighted activations into a heatmap.

    Usage:
        cam = GradCAM(model, model.backbone.layer4)
        heatmap = cam(image_tensor, class_idx=2)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image: torch.Tensor, class_idx: int) -> torch.Tensor:
        """
        Args:
            image: (1, C, H, W) tensor, already on the model's device.
            class_idx: which output class to explain.

        Returns:
            (H, W) heatmap tensor, values normalized to [0, 1].
        """
        self.model.zero_grad()
        logits = self.model(image)
        score = logits[0, class_idx]
        score.backward()

        # weight each activation channel by how much it mattered for this class
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        weighted_activations = (weights * self.activations).sum(dim=1, keepdim=True)
        heatmap = F.relu(weighted_activations)

        heatmap = F.interpolate(heatmap, size=image.shape[2:], mode="bilinear", align_corners=False)
        heatmap = heatmap.squeeze().cpu()
        heatmap -= heatmap.min()
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap


def overlay_heatmap(image: torch.Tensor, heatmap: torch.Tensor, alpha: float = 0.4) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap onto the original (normalized) image tensor.

    Args:
        image: (3, H, W) tensor, normalized with ImageNet mean/std.
        heatmap: (H, W) tensor, values in [0, 1] (output of GradCAM.__call__).
        alpha: heatmap opacity in the blend.

    Returns:
        (H, W, 3) uint8 RGB array, ready to save/plot.
    """
    denorm = image.cpu() * IMAGENET_STD + IMAGENET_MEAN
    img_np = denorm.clamp(0, 1).permute(1, 2, 0).numpy()

    colored = cm.jet(heatmap.numpy())[:, :, :3]
    blended = (1 - alpha) * img_np + alpha * colored
    return (blended * 255).clip(0, 255).astype(np.uint8)
