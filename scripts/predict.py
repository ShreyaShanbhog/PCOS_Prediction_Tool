import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ['PCOS / Infected', 'Non-PCOS / Not Infected']

# -----------------------------
# Load Model
# -----------------------------
def load_model():

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(torch.load("resnet50_baseline.pth", map_location=device))

    model = model.to(device)
    model.eval()

    return model


model = load_model()

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -----------------------------
# Prediction Function
# -----------------------------
def predict(image_file):

    image = Image.open(image_file).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()

    return classes[prediction]
