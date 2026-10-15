from pathlib import Path
from typing import Callable, Sequence

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class ProductDefectDataset(Dataset):
    """PyTorch dataset for product images and binary class labels."""

    def __init__(
        self,
        image_paths: Sequence[Path],
        labels: Sequence[int],
        transform: Callable | None = None,
    ) -> None:
        if len(image_paths) != len(labels):
            raise ValueError(
                "image_paths and labels must contain the same number of items."
            )

        self.image_paths = [Path(path) for path in image_paths]
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        label = self.labels[index]

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load image: {image_path}"
            ) from exc

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def split_dataset(
    image_paths: Sequence[Path],
    labels: Sequence[int],
    val_size: float,
    test_size: float,
    random_seed: int,
):
    """Create stratified train, validation, and test splits."""

    if len(image_paths) != len(labels):
        raise ValueError(
            "image_paths and labels must contain the same number of items."
        )

    if val_size <= 0 or test_size <= 0:
        raise ValueError(
            "val_size and test_size must both be greater than 0."
        )

    if val_size + test_size >= 1:
        raise ValueError(
            "val_size + test_size must be less than 1."
        )

    image_paths = list(image_paths)
    labels = list(labels)

    temp_size = val_size + test_size

    (
        train_paths,
        temp_paths,
        train_labels,
        temp_labels,
    ) = train_test_split(
        image_paths,
        labels,
        test_size=temp_size,
        stratify=labels,
        random_state=random_seed,
    )

    relative_test_size = test_size / temp_size

    (
        val_paths,
        test_paths,
        val_labels,
        test_labels,
    ) = train_test_split(
        temp_paths,
        temp_labels,
        test_size=relative_test_size,
        stratify=temp_labels,
        random_state=random_seed,
    )

    return (
        train_paths,
        val_paths,
        test_paths,
        train_labels,
        val_labels,
        test_labels,
    )
