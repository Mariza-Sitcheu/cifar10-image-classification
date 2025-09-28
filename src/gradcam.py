import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.preprocess import load_cifar10_data
from src.model import CustomCNN, get_resnet18


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks to capture gradients and activations
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, input_image, class_idx=None):
        self.model.eval()
        input_image = input_image.requires_grad_(True)

        # Forward pass
        output = self.model(input_image)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for the target class
        output[:, class_idx].backward()

        # Compute Grad-CAM
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze().cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-10  # Normalize

        return heatmap, class_idx


def visualize_gradcam(model, model_path, testloader, classes, model_name, num_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)

    # Select target layer (last convolutional layer)
    if model_name == "custom_cnn":
        target_layer = model.conv2  # Last conv layer in CustomCNN
    else:
        target_layer = model.layer4[-1]  # Last block in ResNet-18

    grad_cam = GradCAM(model, target_layer)

    # Get sample images
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images[:num_samples].to(device), labels[:num_samples].to(device)

    for i in range(num_samples):
        img = images[i:i + 1]  # Batch of 1
        heatmap, pred_idx = grad_cam.generate(img)

        # Denormalize image for visualization
        img_np = img.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
        img_np = img_np * 0.5 + 0.5  # Denormalize
        img_np = (img_np * 255).astype(np.uint8)

        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (32, 32))
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Overlay heatmap on original image
        overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

        # Plot original, heatmap, and overlay
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(img_np)
        plt.title(f'Original\nTrue: {classes[labels[i]]}')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(heatmap)
        plt.title('Grad-CAM Heatmap')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title(f'Overlay\nPred: {classes[pred_idx]}')
        plt.axis('off')

        plt.suptitle(f'Grad-CAM - {model_name} Sample {i + 1}')
        os.makedirs('figures', exist_ok=True)  # Ensure figures/ exists
        plt.savefig(f'figures/gradcam_{model_name}_sample_{i + 1}.png')
        plt.close()


if __name__ == "__main__":
    # Load data
    _, testloader, classes = load_cifar10_data()

    # Generate Grad-CAM for custom CNN
    custom_cnn = CustomCNN()
    visualize_gradcam(custom_cnn, "models/custom_cnn.pth", testloader, classes, "custom_cnn")

    # Generate Grad-CAM for ResNet-18
    resnet18 = get_resnet18(pretrained=True)
    visualize_gradcam(resnet18, "models/resnet18.pth", testloader, classes, "resnet18")