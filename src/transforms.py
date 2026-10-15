from torchvision import transforms

from src.config import IMAGE_SIZE


# ImageNet normalization used by the pretrained ResNet18 weights.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms():
    """Return preprocessing and mild augmentation for training images."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def get_eval_transforms():
    """Return deterministic preprocessing for validation/test images."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )
