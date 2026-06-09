"""
Grad-CAM implementation for CNN and ResNet models.

Usage:
    gcam    = GradCAM(model, target_layer)
    heatmap = gcam.generate(input_tensor, class_idx)  # returns H×W float32 0–1
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Visualises the regions of an input image most relevant to a model's
    prediction by computing the gradient of the class score with respect
    to the feature maps of a target convolutional layer.

    Reference: Selvaraju et al. (2017) https://arxiv.org/abs/1610.02391

    Args:
        model:        PyTorch model in eval mode.
        target_layer: The convolutional layer to hook into.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._activations: Tensor | None = None
        self._gradients:   Tensor | None = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""
        def save_activation(_, __, output):
            self._activations = output.detach()

        def save_gradient(_, __, grad_output):
            self._gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(save_activation)
        self.target_layer.register_full_backward_hook(save_gradient)

    def generate(self, input_tensor: Tensor, class_idx: int) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for the given class.

        Args:
            input_tensor: Preprocessed input tensor of shape (1, C, H, W).
            class_idx:    Index of the target class.

        Returns:
            Normalised heatmap as a float32 numpy array of shape (H, W),
            values in [0, 1].
        """
        self.model.zero_grad()

        output = self.model(input_tensor)
        score  = output[0, class_idx]
        score.backward()

        # Global average pool the gradients over spatial dimensions
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self._activations).sum(dim=1).squeeze()  # (H, W)
        cam = torch.relu(cam).cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)