"""
WideResNet-16-4 implementation for SVHN.

Based on Zagoruyko & Komodakis (2016), "Wide Residual Networks".
WRN-16-4: depth=16, widen_factor=4.
This is a DeepOBS standard test problem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block with two 3×3 convolutions and optional dropout."""

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1, dropout_rate: float = 0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout_rate = dropout_rate

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    """
    Wide Residual Network.

    Args:
        depth: Total depth of the network (must satisfy (depth-4) % 6 == 0)
        widen_factor: Width multiplier
        num_classes: Number of output classes
        dropout_rate: Dropout probability inside residual blocks
        in_channels: Number of input channels
    """

    def __init__(
        self,
        depth: int = 16,
        widen_factor: int = 4,
        num_classes: int = 10,
        dropout_rate: float = 0.0,
        in_channels: int = 3,
    ):
        super().__init__()
        assert (depth - 4) % 6 == 0, f"Depth must satisfy (depth-4) % 6 == 0, got {depth}"
        n = (depth - 4) // 6  # Number of blocks per group

        nStages = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(in_channels, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(nStages[0], nStages[1], n, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(nStages[1], nStages[2], n, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(nStages[2], nStages[3], n, stride=2, dropout_rate=dropout_rate)

        self.bn = nn.BatchNorm2d(nStages[3])
        self.fc = nn.Linear(nStages[3], num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_planes, out_planes, num_blocks, stride, dropout_rate):
        layers = [BasicBlock(in_planes, out_planes, stride, dropout_rate)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_planes, out_planes, 1, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
