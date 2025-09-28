import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.preprocess import load_cifar10_data
from src.model import CustomCNN, get_resnet18

def evaluate_model(model, model_path, testloader, classes, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, weights_only=True))  # Updated to weights_only=True
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy of {model_name} on test images: {accuracy:.2f}%')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure figures directory exists
    os.makedirs('figures', exist_ok=True)  # Create directory if it doesn't exist
    plt.savefig(f'figures/confusion_matrix_{model_name}.png')
    plt.close()

if __name__ == "__main__":
    # Load data
    _, testloader, classes = load_cifar10_data()

    # Evaluate custom CNN
    custom_cnn = CustomCNN()
    evaluate_model(custom_cnn, "models/custom_cnn.pth", testloader, classes, "custom_cnn")

    # Evaluate ResNet-18
    resnet18 = get_resnet18(pretrained=True)
    evaluate_model(resnet18, "models/resnet18.pth", testloader, classes, "resnet18")