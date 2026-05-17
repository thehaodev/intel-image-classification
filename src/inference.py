from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from torchvision import transforms

from src.model_registry import CLASS_NAMES


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class PredictionResult:
    model_name: str
    display_name: str
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def predict_image(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    model_name: str,
    display_name: str,
    image_size: int = 224,
) -> PredictionResult:
    transform = build_eval_transform(image_size)
    rgb_image = image.convert("RGB")
    inputs = transform(rgb_image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()

    confidence, class_idx = torch.max(probs, dim=0)
    probabilities = {class_name: float(probs[idx]) for idx, class_name in enumerate(CLASS_NAMES)}

    return PredictionResult(
        model_name=model_name,
        display_name=display_name,
        predicted_class=CLASS_NAMES[int(class_idx)],
        confidence=float(confidence),
        probabilities=probabilities,
    )
