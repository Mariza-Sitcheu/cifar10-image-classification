"""
Post-hoc temperature scaling calibration.

Learns a single temperature parameter T for each model by minimising
the negative log-likelihood on the validation set. T is saved to
src/models/<model_name>_temperature.json.

Usage:
    python -m src.calibrate
"""

import json
import os
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch import Tensor

from src.preprocess import load_cifar10_data
from src.model import CustomCNN, get_resnet18


# -- Temperature scaling---------------------------------------------------

class TemperatureScaler(nn.Module):
    """
    Wraps a trained model and scales its logits by a learned temperature T.

    Only T is optimised — the base model weights are frozen.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))  # start at T=1

    def forward(self, x: Tensor) -> Tensor:
        logits = self.model(x)
        return logits / self.temperature


def calibrate(model: nn.Module, val_loader, device: torch.device) -> float:
    """
    Learn the optimal temperature T on the validation set.

    Args:
        model:      Trained model in eval mode.
        val_loader: DataLoader for the validation/test set.
        device:     CPU or CUDA device.

    Returns:
        Optimal temperature T as a float.
    """
    scaler = TemperatureScaler(model).to(device)

    # Freeze the base model — only T is trainable
    for param in scaler.model.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS(
        [scaler.temperature], lr=0.01, max_iter=50
    )

    # Collect all logits and labels from the validation set
    all_logits, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # LBFGS requires a closure that recomputes the loss each step
    def closure():
        optimizer.zero_grad()
        loss = criterion(all_logits.to(device) / scaler.temperature, all_labels.to(device))
        loss.backward()
        return loss

    optimizer.step(closure)

    T = scaler.temperature.item()
    return T


# -- Main -------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, _ = load_cifar10_data()

    models_to_calibrate = {
        "custom_cnn": CustomCNN(),
        "resnet18":   get_resnet18(pretrained=False),
    }

    for model_name, model in models_to_calibrate.items():
        weights_path = f"src/models/{model_name}_best.pth"

        if not os.path.exists(weights_path):
            print(f"Skipping {model_name} — weights not found at {weights_path}")
            continue

        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()

        print(f"Calibrating {model_name}...")
        T = calibrate(model, val_loader, device)
        print(f"  Optimal temperature: T = {T:.4f}")

        # Save T to disk
        out_path = f"src/models/{model_name}_temperature.json"
        with open(out_path, "w") as f:
            json.dump({"temperature": T}, f)
        print(f"  Saved to {out_path}")
