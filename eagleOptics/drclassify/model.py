import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

class ResNet18CNN(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18CNN, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def load_model():
    model = ResNet18CNN(num_classes=5)  # Adjust num_classes to match your problem
    model.load_state_dict(torch.load('./ResNet18_CNN_Model.pth', map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model
