import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import os
import numpy as np

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# Dataset
# -----------------------------
dataset = datasets.ImageFolder("data/train", transform=transform)

# 80/20 split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
_, test_dataset = random_split(dataset, [train_size, test_size])

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

classes = dataset.classes
print("Classes:", classes)
print("Total test samples:", len(test_dataset))

# -----------------------------
# Load Saved Model
# -----------------------------
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("resnet50_baseline.pth", map_location=device))
model = model.to(device)
model.eval()
print("Model loaded successfully!")

# -----------------------------
# Evaluate on Test Set
# -----------------------------
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/confusion_matrix.png")
print("Confusion matrix plot saved as plots/confusion_matrix.png")

# -----------------------------
# Classification Report
# -----------------------------
report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
print("Classification Report:\n", classification_report(all_labels, all_preds, target_names=classes))

df_report = pd.DataFrame(report).transpose()
df_report.to_csv("plots/classification_report.csv", index=True)
print("Classification report saved as plots/classification_report.csv")

# -----------------------------
# Accuracy Plot (single point since no per-epoch loss)
# -----------------------------
accuracy = 100 * sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
plt.figure()
plt.bar(["Test Accuracy"], [accuracy], color='green')
plt.ylim(0, 100)
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy")
plt.grid(axis='y')
plt.savefig("plots/test_accuracy.png")
print(f"Test accuracy plot saved as plots/test_accuracy.png ({accuracy:.2f}%)")
