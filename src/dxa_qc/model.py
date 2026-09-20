"""U-Net с энкодером ResNet-34: по каналу heatmap на каждую точку.

Энкодер реализован здесь, а веса ImageNet подгружаются из torchvision, если он доступен.
Так архитектура не зависит от библиотеки, а вес — опционален.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, 3, stride, 1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1, self.bn1 = conv3(cin, cout, stride), nn.BatchNorm2d(cout)
        self.conv2, self.bn2 = conv3(cout, cout), nn.BatchNorm2d(cout)
        self.downsample = (
            nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
            if stride != 1 or cin != cout else None
        )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(self.bn2(self.conv2(x)) + identity, inplace=True)


class ResNet34Encoder(nn.Module):
    """Пять уровней: /2, /4, /8, /16, /32. Имена слоёв совпадают с torchvision."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make(64, 64, 3, 1)
        self.layer2 = self._make(64, 128, 4, 2)
        self.layer3 = self._make(128, 256, 6, 2)
        self.layer4 = self._make(256, 512, 3, 2)
        self.out_channels = (64, 64, 128, 256, 512)

    @staticmethod
    def _make(cin, cout, blocks, stride):
        layers = [BasicBlock(cin, cout, stride)]
        layers += [BasicBlock(cout, cout) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = F.relu(self.bn1(self.conv1(x)), inplace=True)   # /2
        x1 = self.layer1(self.maxpool(x0))                   # /4
        x2 = self.layer2(x1)                                 # /8
        x3 = self.layer3(x2)                                 # /16
        x4 = self.layer4(x3)                                 # /32
        return [x0, x1, x2, x3, x4]

    def load_imagenet(self) -> bool:
        try:
            from torchvision.models import ResNet34_Weights, resnet34
        except ImportError:
            return False
        src = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).state_dict()
        own = self.state_dict()
        src = {k: v for k, v in src.items() if k in own and v.shape == own[k].shape}
        self.load_state_dict(src, strict=False)
        return True


class DecoderBlock(nn.Module):
    def __init__(self, cin, skip, cout):
        super().__init__()
        self.block = nn.Sequential(
            conv3(cin + skip, cout), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            conv3(cout, cout), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


class KeypointNet(nn.Module):
    """Вход 1×H×W (H, W кратны 32), выход n_channels×H×W — логиты heatmap."""

    def __init__(self, n_channels: int, decoder_channels=(256, 128, 64, 32, 16),
                 pretrained: bool = True):
        super().__init__()
        self.encoder = ResNet34Encoder()
        self.pretrained_loaded = self.encoder.load_imagenet() if pretrained else False
        enc = self.encoder.out_channels                      # (64, 64, 128, 256, 512)
        skips = (enc[3], enc[2], enc[1], enc[0], 0)
        cin = enc[4]
        blocks = []
        for skip, cout in zip(skips, decoder_channels):
            blocks.append(DecoderBlock(cin, skip, cout))
            cin = cout
        self.decoder = nn.ModuleList(blocks)
        self.head = nn.Conv2d(decoder_channels[-1], n_channels, 1)
        nn.init.constant_(self.head.bias, -4.0)              # старт с «молчания»

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)                         # серый → 3 канала ImageNet
        feats = self.encoder(x)
        y = feats[-1]
        for block, skip in zip(self.decoder, [feats[3], feats[2], feats[1], feats[0], None]):
            y = block(y, skip)
        return self.head(y)

    def param_groups(self, lr_encoder: float, lr_decoder: float):
        return [
            {"params": self.encoder.parameters(), "lr": lr_encoder},
            {"params": list(self.decoder.parameters()) + list(self.head.parameters()),
             "lr": lr_decoder},
        ]
