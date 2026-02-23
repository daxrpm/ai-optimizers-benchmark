"""SimpleCNN for Fashion-MNIST — DeepOBS-style 2-conv + 2-FC architecture."""

import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple CNN for 28×28 grayscale images.

    Architecture (matches DeepOBS F-MNIST CNN):
      Conv(1, 32, 5) → ReLU → MaxPool(2)
      Conv(32, 64, 5) → ReLU → MaxPool(2)
      FC(1024, 256) → ReLU → Dropout(0.5)
      FC(256, num_classes)
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
