"""
Unit tests for model inference and Grad-CAM.

Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import torch
import torchvision.models as tv_models
from PIL import Image

from src.model import CustomCNN
from src.gradcam import GradCAM


# -- Fixtures -------------------------------------------------------

@pytest.fixture
def custom_cnn() -> CustomCNN:
    """Untrained CustomCNN in eval mode."""
    model = CustomCNN()
    model.eval()
    return model


@pytest.fixture
def resnet18() -> torch.nn.Module:
    """Untrained ResNet-18 with 10-class head in eval mode."""
    model = tv_models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.eval()
    return model


@pytest.fixture
def dummy_input() -> torch.Tensor:
    """Batch of 1 random CIFAR-10 sized image tensor."""
    return torch.randn(1, 3, 32, 32)


# -- CustomCNN tests --------------------------------------------

def test_custom_cnn_output_shape(custom_cnn, dummy_input):
    """CustomCNN must output (batch, 10) logits."""
    with torch.no_grad():
        output = custom_cnn(dummy_input)
    assert output.shape == (1, 10), f"Expected (1, 10), got {output.shape}"


def test_custom_cnn_output_is_finite(custom_cnn, dummy_input):
    """CustomCNN output should not contain NaN or Inf."""
    with torch.no_grad():
        output = custom_cnn(dummy_input)
    assert torch.isfinite(output).all()


def test_custom_cnn_softmax_sums_to_one(custom_cnn, dummy_input):
    """Softmax probabilities must sum to 1."""
    with torch.no_grad():
        logits = custom_cnn(dummy_input)
        probs = torch.softmax(logits, dim=1)
    assert abs(probs.sum().item() - 1.0) < 1e-5


# -- ResNet-18 tests --------------------------------------------

def test_resnet18_output_shape(resnet18, dummy_input):
    """ResNet-18 with custom head must output (batch, 10) logits."""
    with torch.no_grad():
        output = resnet18(dummy_input)
    assert output.shape == (1, 10)


def test_resnet18_output_is_finite(resnet18, dummy_input):
    """ResNet-18 output should not contain NaN or Inf."""
    with torch.no_grad():
        output = resnet18(dummy_input)
    assert torch.isfinite(output).all()


# -- Grad-CAM tests --------------------------------------------

def test_gradcam_output_shape(custom_cnn, dummy_input):
    """Grad-CAM heatmap must be 2D."""
    target_layer = custom_cnn.conv2
    gcam = GradCAM(custom_cnn, target_layer)
    heatmap = gcam.generate(dummy_input, class_idx=0)

    assert heatmap.ndim == 2, "Heatmap must be 2D"
    assert heatmap.dtype == np.float32


def test_gradcam_values_in_range(custom_cnn, dummy_input):
    """Grad-CAM values must be normalised to [0, 1]."""
    target_layer = custom_cnn.conv2
    gcam = GradCAM(custom_cnn, target_layer)
    heatmap = gcam.generate(dummy_input, class_idx=3)

    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_gradcam_resnet(resnet18, dummy_input):
    """Grad-CAM must work on ResNet-18 targeting layer4[-1].conv2.

    Note: ResNet-18 downsamples 32x32 CIFAR input to a 1x1 feature map by
    layer4, so the heatmap will be (1, 1) — still 2D, just very coarse.
    """
    target_layer = resnet18.layer4[-1].conv2
    gcam = GradCAM(resnet18, target_layer)
    heatmap = gcam.generate(dummy_input, class_idx=5)

    assert heatmap.ndim == 2, f"Expected 2D heatmap, got shape {heatmap.shape}"
    assert 0.0 <= heatmap.max() <= 1.0


# -- Integration test --------------------------------------------

def test_top3_predictions_sum_less_than_one(custom_cnn, dummy_input):
    """Top-3 softmax probabilities must sum to at most 1."""
    with torch.no_grad():
        logits = custom_cnn(dummy_input)
        probs = torch.softmax(logits, dim=1).squeeze()
        top3_sum = torch.topk(probs, 3).values.sum().item()

    assert top3_sum <= 1.0 + 1e-5
