from __future__ import annotations
import torch
from torch import nn
from torchvision import models
from .adacos import AdaCos


class EfficientNetB3OneChannel(nn.Module):
    def __init__(self, imagenet_init: bool = False):
        super().__init__()
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if imagenet_init else None
        model = models.efficientnet_b3(weights=weights)
        conv = model.features[0][0]
        if conv.in_channels != 1:
            new_conv = nn.Conv2d(
                1, conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                bias=False,
            )
            if imagenet_init:
                with torch.no_grad():
                    new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
            model.features[0][0] = new_conv
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.num_features = 1536

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class Network(nn.Module):
    def __init__(self, imagenet_init: bool = False):
        super().__init__()
        self.in_channels = 1
        self.backbone = EfficientNetB3OneChannel(imagenet_init=imagenet_init)
        self.num_features = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class Featurier(nn.Module):
    """Name retained for checkpoint compatibility with archived code."""
    def __init__(self, imagenet_init: bool = False, px_size=(320, 320)):
        super().__init__()
        model = Network(imagenet_init=imagenet_init)
        self.backbone = model.backbone
        self.px_size = tuple(px_size)
        self.num_features = model.num_features
        self.in_channels = model.in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class Classifier(nn.Module):
    def __init__(self, num_features: int, n_classes: int, hidden_dim: int = 1280, dropout: float = 0.5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_features),
        )
        self.metric = AdaCos(num_features=num_features, num_classes=n_classes)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        return self.metric(self.fc(x), labels)


class CNN(nn.Module):
    """EfficientNet-B3 + projection head + AdaCos.

    Module names mirror the recovered checkpoint so `strict=True` loading can
    be used as a structural integrity check.
    """
    def __init__(
        self,
        imagenet_init: bool = False,
        px_size=(320, 320),
        n_classes: int = 30805,
        hidden_dim: int = 1280,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.featurier = Featurier(imagenet_init=imagenet_init, px_size=px_size)
        self.classifier = Classifier(
            num_features=self.featurier.num_features,
            n_classes=n_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.num_features = self.featurier.num_features
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        return self.classifier(self.featurier(x), labels)

    def encode(self, x: torch.Tensor, mode: str = "paper1280") -> torch.Tensor:
        """Return an embedding without AdaCos classification.

        Modes
        -----
        paper1280:
            1280-D bottleneck after Linear(1536->1280) + ReLU. This is the
            public paper-compatible interpretation of the reported 1280-D embedding.
        preadacos1536:
            Full projection-head output immediately before AdaCos.
        backbone1536:
            Pooled EfficientNet-B3 output used by some recovered later scripts.
        """
        b = self.featurier(x)
        if mode == "backbone1536":
            return b
        h = self.classifier.fc[0](b)
        h = self.classifier.fc[1](h)
        if mode == "paper1280":
            return h
        h = self.classifier.fc[2](h)
        h = self.classifier.fc[3](h)
        if mode == "preadacos1536":
            return h
        raise ValueError(f"Unknown embedding mode: {mode}")
