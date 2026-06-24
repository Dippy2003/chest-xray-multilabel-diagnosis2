"""
Grad-CAM for the ResNet50 transfer model - shows which pixels the model
actually looked at for a given predicted class, instead of trusting the
AUC/F1 numbers blindly.
"""
import torch
import torch.nn.functional as F


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
