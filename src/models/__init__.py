"""Model factory and registry."""

import torch.nn as nn
from typing import Any

from src.models.simple_cnn import SimpleCNN
from src.models.resnet import resnet18, resnet34
from src.models.wrn import WideResNet
from src.models.mlp import MLP


def create_model(name: str, **kwargs: Any) -> nn.Module:
    """
    Create a model by name.

    Supported models:
      - 'simple_cnn': SimpleCNN for Fashion-MNIST
      - 'resnet18': ResNet-18 for CIFAR-10
      - 'resnet34': ResNet-34 for CIFAR-100
      - 'wrn164': WideResNet-16-4 for SVHN
      - 'mlp': MLP for tabular tasks
    """
    name = name.lower()

    if name == "simple_cnn":
        return SimpleCNN(
            num_classes=kwargs.get("num_classes", 10),
            in_channels=kwargs.get("in_channels", 1),
        )
    elif name == "resnet18":
        return resnet18(
            num_classes=kwargs.get("num_classes", 10),
            pretrained=kwargs.get("pretrained", False),
        )
    elif name == "resnet34":
        return resnet34(
            num_classes=kwargs.get("num_classes", 100),
            pretrained=kwargs.get("pretrained", False),
        )
    elif name == "wrn164":
        return WideResNet(
            depth=16,
            widen_factor=4,
            num_classes=kwargs.get("num_classes", 10),
            dropout_rate=kwargs.get("dropout_rate", 0.0),
            in_channels=kwargs.get("in_channels", 3),
        )
    elif name == "mlp":
        return MLP(
            input_dim=kwargs["input_dim"],
            hidden_dims=kwargs.get("hidden_dims", [256, 128, 64]),
            num_classes=kwargs.get("num_classes", 2),
            dropout_rate=kwargs.get("dropout_rate", 0.3),
            use_batchnorm=kwargs.get("use_batchnorm", True),
        )
    else:
        raise ValueError(f"Unknown model: {name}. Choose from: simple_cnn, resnet18, resnet34, wrn164, mlp")
