from torch import nn
from torchvision import models

class CephNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ResNet = models.resnet50(weights="DEFAULT")
        # features from original fc later
        num_features = self.ResNet.fc.in_features
        # Re-assign ResNet FC layer
        self.ResNet.fc = nn.Linear (num_features, 58)
        # Number of flattened features (b, 58)

    def forward(self, x):
        # (b,58)
        flat_output = self.ResNet(x)
        # (1, 29, 2)
        output = flat_output.view(-1,29,2)

        return output

