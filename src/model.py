import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from src.config import NUM_CLASSES


def create_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Create a ResNet18 model for binary product classification."""

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)

    return model
