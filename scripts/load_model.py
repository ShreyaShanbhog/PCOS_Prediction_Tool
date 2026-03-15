import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate same architecture
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

# Load saved weights
model.load_state_dict(torch.load("resnet50_baseline.pth"))
model = model.to(device)

model.eval()
print("Model loaded successfully!")
