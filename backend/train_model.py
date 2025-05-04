import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from pathlib import Path
import requests
import os
from tqdm import tqdm

def download_food101():
    """Download Food101 dataset if not already present"""
    print("Checking/Downloading Food101 dataset...")
    return torchvision.datasets.Food101(
        root='./data',
        split='train',
        download=True
    )

def setup_data_loaders():
    """Set up data loaders for training and validation"""
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load datasets
    train_dataset = torchvision.datasets.Food101(
        root='./data',
        split='train',
        transform=transform,
        download=True
    )

    val_dataset = torchvision.datasets.Food101(
        root='./data',
        split='test',
        transform=transform,
        download=True
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader, train_dataset.classes

def train_model(model, train_loader, val_loader, device, num_epochs=10):
    """Train the model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100. * correct / total
        val_loss = val_loss / len(val_loader)

        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')

        # Save best model
        if val_acc > best_acc:
            print('Saving model...')
            torch.save(model.state_dict(), 'food_model_weights.pth')
            best_acc = val_acc

        scheduler.step(val_loss)

def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Download dataset
    download_food101()

    # Set up data loaders
    train_loader, val_loader, classes = setup_data_loaders()

    # Initialize model
    model = resnet50(pretrained=True)
    num_classes = len(classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Train model
    train_model(model, train_loader, val_loader, device)

    print('Training completed!')

if __name__ == '__main__':
    main() 