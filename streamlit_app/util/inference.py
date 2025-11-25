import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

class LandmarkClassifier:
    def __init__(self, model_path, class_names):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained ResNet18
        self.model = models.resnet18(weights=None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, len(class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.class_names = class_names

        # Transform must match training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_path):
        """Takes an image path and returns predicted class name"""

        image = Image.open(image_path).convert("RGB")
        img = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            _, predicted = torch.max(outputs, 1)

        return self.class_names[predicted.item()]
