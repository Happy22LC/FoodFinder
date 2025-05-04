import os
import shutil
import random
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

# List of famous dishes to include
FAMOUS_DISHES = [
    'pizza', 'hamburger', 'sushi', 'steak', 'tacos', 'ramen', 'spaghetti_bolognese',
    'caesar_salad', 'pancakes', 'ice_cream', 'fried_rice', 'hot_dog', 'nachos',
    'chicken_wings', 'dumplings', 'falafel', 'paella', 'waffles', 'tiramisu', 'fish_and_chips',
]

# Updated paths to use correct directory structure
DATA_DIR = os.path.join('backend', 'data', 'food-101', 'images')
DST_ROOT = os.path.join('backend', 'data', 'food-20-famous')
IMAGES_PER_DISH = 150

print(f"Creating dataset from {DATA_DIR} to {DST_ROOT}")
os.makedirs(DST_ROOT, exist_ok=True)

for dish in FAMOUS_DISHES:
    src_dir = os.path.join(DATA_DIR, dish)
    dst_dir = os.path.join(DST_ROOT, dish)
    print(f"Processing {dish}...")
    os.makedirs(dst_dir, exist_ok=True)
    images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    selected = random.sample(images, min(IMAGES_PER_DISH, len(images)))
    for img in selected:
        shutil.copy2(os.path.join(src_dir, img), os.path.join(dst_dir, img))

print(f"Subset created at {DST_ROOT} with {IMAGES_PER_DISH} images per dish.")

# Paths
data_dir = DST_ROOT

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset and DataLoader
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model (using pretrained ResNet18)
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (simple version)
for epoch in range(5):  # Increase epochs for better results
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

# Save model
torch.save(model.state_dict(), 'food20_model.pth') 