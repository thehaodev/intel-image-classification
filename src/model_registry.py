from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, resnet50, swin_t, vit_b_16


CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def replace_classifier(model: nn.Module, model_name: str, num_classes: int, dropout: float) -> nn.Module:
    if model_name == "resnet50":
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    elif model_name == "mobilenet_v2":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name == "vit_b_16":
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    elif model_name == "swin_t":
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return model


def build_model(model_name: str, num_classes: int = len(CLASS_NAMES), dropout: float = 0.3) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, dropout=dropout)
    if model_name == "resnet50":
        return replace_classifier(resnet50(weights=None), model_name, num_classes, dropout)
    if model_name == "mobilenet_v2":
        return replace_classifier(mobilenet_v2(weights=None), model_name, num_classes, dropout)
    if model_name == "vit_b_16":
        return replace_classifier(vit_b_16(weights=None), model_name, num_classes, dropout)
    if model_name == "swin_t":
        return replace_classifier(swin_t(weights=None), model_name, num_classes, dropout)

    valid_names = ["simple_cnn", "resnet50", "mobilenet_v2", "vit_b_16", "swin_t"]
    raise ValueError(f"model_name must be one of: {valid_names}")


def load_checkpoint_model(
    model_name: str,
    checkpoint_path: str | Path,
    device: torch.device,
    num_classes: int = len(CLASS_NAMES),
) -> nn.Module:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(model_name=model_name, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
