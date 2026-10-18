from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torchvision.models import ResNet18_Weights, resnet18


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalGenerator(nn.Module):
    def __init__(
        self,
        z_dim: int = 128,
        num_classes: int = 2,
        base_channels: int = 64,
        image_size: int = 128,
    ) -> None:
        super().__init__()
        if image_size != 128:
            raise ValueError("This reference generator is configured for 128x128 images")
        self.z_dim = int(z_dim)
        self.num_classes = int(num_classes)
        self.base_channels = int(base_channels)
        self.image_size = int(image_size)
        self.label_embedding = nn.Embedding(num_classes, z_dim)
        self.fc = nn.Linear(z_dim * 2, base_channels * 16 * 4 * 4)
        self.blocks = nn.Sequential(
            GeneratorBlock(base_channels * 16, base_channels * 8),
            GeneratorBlock(base_channels * 8, base_channels * 4),
            GeneratorBlock(base_channels * 4, base_channels * 2),
            GeneratorBlock(base_channels * 2, base_channels),
            GeneratorBlock(base_channels, base_channels // 2),
        )
        self.to_image = nn.Sequential(
            nn.Conv2d(base_channels // 2, 1, 3, padding=1),
            nn.Tanh(),
        )
        self.apply(init_gan_weights)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embedded = self.label_embedding(labels)
        hidden = self.fc(torch.cat([noise, embedded], dim=1))
        hidden = hidden.view(noise.size(0), self.base_channels * 16, 4, 4)
        return self.to_image(self.blocks(hidden))


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ProjectionDiscriminator(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        base_channels: int = 64,
        image_size: int = 128,
    ) -> None:
        super().__init__()
        if image_size != 128:
            raise ValueError("This reference discriminator is configured for 128x128 images")
        self.num_classes = int(num_classes)
        self.base_channels = int(base_channels)
        self.image_size = int(image_size)
        self.blocks = nn.Sequential(
            DiscriminatorBlock(1, base_channels),
            DiscriminatorBlock(base_channels, base_channels * 2),
            DiscriminatorBlock(base_channels * 2, base_channels * 4),
            DiscriminatorBlock(base_channels * 4, base_channels * 8),
            DiscriminatorBlock(base_channels * 8, base_channels * 16),
        )
        feature_dim = base_channels * 16
        self.unconditional = spectral_norm(nn.Linear(feature_dim, 1))
        self.label_embedding = spectral_norm(nn.Embedding(num_classes, feature_dim))
        self.apply(init_gan_weights)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        hidden = self.blocks(images)
        hidden = hidden.sum(dim=(2, 3))
        unconditional = self.unconditional(hidden).squeeze(1)
        projection = (self.label_embedding(labels) * hidden).sum(dim=1)
        return unconditional + projection


def init_gan_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear, nn.Embedding)):
        weight = getattr(module, "weight_orig", None)
        if weight is None:
            weight = getattr(module, "weight", None)
        if weight is not None:
            nn.init.normal_(weight, mean=0.0, std=0.02)
    if isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias)


class CXRFeatureClassifier(nn.Module):
    """Binary classifier whose 256-D projection is used as the medical FD space."""

    def __init__(
        self,
        feature_dim: int = 256,
        num_classes: int = 2,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.feature_dim = int(feature_dim)
        self.register_buffer(
            "imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _prepare_input(self, images: torch.Tensor) -> torch.Tensor:
        # Dataset and generator both use [-1, 1]. Keep this operation differentiable.
        images = (images + 1.0) / 2.0
        images = images.clamp(0.0, 1.0)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        if images.shape[-2:] != (224, 224):
            images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        return (images - self.imagenet_mean) / self.imagenet_std

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.projector(self.backbone(self._prepare_input(images)))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(images))


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def load_feature_classifier(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[CXRFeatureClassifier, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_dim = int(checkpoint.get("feature_dim", 256))
    model = CXRFeatureClassifier(feature_dim=feature_dim, pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    return model, checkpoint


def build_gan_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[ConditionalGenerator, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    generator = ConditionalGenerator(
        z_dim=int(config["z_dim"]),
        num_classes=2,
        base_channels=int(config["base_channels"]),
        image_size=int(config["image_size"]),
    )
    generator.load_state_dict(checkpoint["generator"])
    generator.to(device).eval()
    return generator, checkpoint
