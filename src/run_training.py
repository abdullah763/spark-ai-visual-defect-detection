import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    BEST_MODEL_PATH,
    LEARNING_RATE,
    METRICS_DIR,
    NUM_EPOCHS,
    RANDOM_SEED,
)
from src.dataset import ProductDefectDataset, split_dataset
from src.model import create_model
from src.train import (
    save_checkpoint,
    train_one_epoch,
    validate_one_epoch,
)
from src.transforms import (
    get_eval_transforms,
    get_train_transforms,
)
from src.utils import (
    create_project_directories,
    get_device,
    set_seed,
)


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def find_images(folder: Path) -> list[Path]:
    """Find supported image files recursively."""

    if not folder.exists():
        raise FileNotFoundError(
            f"Class directory does not exist: {folder}"
        )

    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def build_image_list(
    normal_dir: Path,
    defective_dir: Path,
):
    """Build image paths and binary labels."""

    normal_images = find_images(normal_dir)
    defective_images = find_images(defective_dir)

    image_paths = normal_images + defective_images

    labels = (
        [0] * len(normal_images)
        + [1] * len(defective_images)
    )

    return image_paths, labels


def save_training_history(history: list[dict]) -> None:
    """Save epoch-level metrics to CSV."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = METRICS_DIR / "training_history.csv"

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys(),
        )

        writer.writeheader()
        writer.writerows(history)

    print(f"Training history saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the product defect classifier."
    )

    parser.add_argument(
        "--normal-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--defective-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    create_project_directories()

    device = get_device()

    print(f"Device: {device}")

    image_paths, labels = build_image_list(
        normal_dir=args.normal_dir,
        defective_dir=args.defective_dir,
    )

    normal_count = labels.count(0)
    defective_count = labels.count(1)

    print(f"Normal images: {normal_count}")
    print(f"Defective images: {defective_count}")
    print(f"Total images: {len(labels)}")

    (
        train_paths,
        val_paths,
        test_paths,
        train_labels,
        val_labels,
        test_labels,
    ) = split_dataset(
        image_paths=image_paths,
        labels=labels,
        val_size=args.val_size,
        test_size=args.test_size,
        random_seed=RANDOM_SEED,
    )

    print(f"Train images: {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")
    print(f"Test images: {len(test_paths)}")

    train_dataset = ProductDefectDataset(
        image_paths=train_paths,
        labels=train_labels,
        transform=get_train_transforms(),
    )

    val_dataset = ProductDefectDataset(
        image_paths=val_paths,
        labels=val_labels,
        transform=get_eval_transforms(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = create_model(
        pretrained=True,
        freeze_backbone=False,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_accuracy = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                model=model,
                checkpoint_path=BEST_MODEL_PATH,
            )

            print(
                f"Saved best model: {BEST_MODEL_PATH}"
            )

    save_training_history(history)

    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
