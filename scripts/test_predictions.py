import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
import os

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Load Dataset Classes
# -----------------------------
# We need the same ImageFolder structure to get class names
train_dataset = datasets.ImageFolder("data/train")
classes = train_dataset.classes
print("Classes:", classes)

# -----------------------------
# Load Model
# -----------------------------
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Replace final layer for 2 classes
model.fc = nn.Linear(model.fc.in_features, 2)

# Load trained weights
model.load_state_dict(torch.load("resnet50_baseline.pth", map_location=device))
model = model.to(device)
model.eval()
print("Model loaded successfully!")

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# Test Predictions on Few Images
# -----------------------------
# Replace this list with paths to any images you want to test
test_images = [
    "data/train/infected/img1.jpg",
    "data/train/notinfected/img1.jpeg",
]

for img_path in test_images:
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue

    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        predicted_class = classes[pred.item()]

    print(f"Image: {img_path} → Predicted class: {predicted_class}")
