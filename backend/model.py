import torch.nn as nn
from torchvision import models

class EmotionResNet(nn.Module):
    def __init__(self):
        super(EmotionResNet, self).__init__()

        self.model = models.resnet18(weights=None)

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 7)

    def forward(self, x):
        return self.model(x)