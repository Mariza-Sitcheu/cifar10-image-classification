import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import os
from tqdm import tqdm
from src.preprocess import load_cifar10_data
from src.model import CustomCNN, get_resnet18

def train_model(model, trainloader, testloader, model_name, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()  # For mixed precision training

    train_losses, val_losses, val_accuracies = [], [], []
    best_val_acc = 0.0
    os.makedirs('models', exist_ok=True)  # Ensure models/ exists

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(trainloader, desc=f'Epoch {epoch + 1}/{epochs} [Train]')

        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Mixed precision training
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            train_bar.set_postfix({'loss': loss.item()})

        train_losses.append(running_loss / len(trainloader))

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_bar = tqdm(testloader, desc=f'Epoch {epoch + 1}/{epochs} [Val]')

        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_bar.set_postfix({'val_loss': loss.item()})

        val_losses.append(val_loss / len(testloader))
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, '
              f'Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            try:
                torch.save(model.state_dict(), f'models/{model_name}_best.pth')
                print(f'Saved best model for {model_name} with Val Acc: {val_acc:.2f}%')
            except Exception as e:
                print(f'Error saving model: {e}')

    # Save final model
    try:
        torch.save(model.state_dict(), f'models/{model_name}.pth')
    except Exception as e:
        print(f'Error saving final model: {e}')

    return train_losses, val_losses, val_accuracies

if __name__ == "__main__":
    # Load data
    trainloader, testloader, classes = load_cifar10_data()

    # Train custom CNN
    custom_cnn = CustomCNN()
    train_model(custom_cnn, trainloader, testloader, "custom_cnn")

    # Train ResNet-18
    resnet18 = get_resnet18(pretrained=True)
    train_model(resnet18, trainloader, testloader, "resnet18")