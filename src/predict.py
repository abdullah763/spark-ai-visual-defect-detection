from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from src.config import CLASS_NAMES, NUM_CLASSES
from src.model import create_model
from src.transforms import get_eval_transforms


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """Load a trained model checkpoint for inference."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    model = create_model(
        num_classes=NUM_CLASSES,
        pretrained=False,
    )

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def predict_image(
    model: nn.Module,
    image: Image.Image,
    device: torch.device,
) -> dict:
    """Predict the class and confidence for a single image."""

    transform = get_eval_transforms()

    image = image.convert("RGB")
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

    confidence, predicted_index = torch.max(
        probabilities,
        dim=1,
    )

    predicted_index = predicted_index.item()
    confidence = confidence.item()

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": confidence,
    }
