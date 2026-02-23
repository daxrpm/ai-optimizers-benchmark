"""ResNet wrappers using torchvision official implementations."""

import torch.nn as nn
import torchvision.models as models


def resnet18(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """
    ResNet-18 with configurable number of output classes.

    For CIFAR-10/100, we modify the first conv layer (no 7×7 stem)
    and remove the max pool to handle 32×32 images properly.
    """
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)

    # Adapt for CIFAR (32×32): replace 7×7 conv with 3×3, remove maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # Replace final FC
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def resnet34(num_classes: int = 100, pretrained: bool = False) -> nn.Module:
    """
    ResNet-34 with configurable number of output classes.
    Adapted for CIFAR-sized inputs (32×32).
    """
    if pretrained:
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet34(weights=None)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
