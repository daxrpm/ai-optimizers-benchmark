"""Configurable MLP for tabular data tasks."""

import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for tabular classification/regression.

    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions
        num_classes: Number of output classes
        dropout_rate: Dropout probability between layers
        use_batchnorm: Whether to use BatchNorm after each hidden layer
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [256, 128, 64],
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)
