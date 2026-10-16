import argparse
import csv
import json
import shutil
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    ERROR_ANALYSIS_DIR,
    METRICS_DIR,
    PLOTS_DIR,
    RANDOM_SEED,
)
from src.dataset import ProductDefectDataset, split_dataset
from src.evaluate import (
    evaluate_model,
    get_error_indices,
    save_confusion_matrix,
)
from src.predict import load_model, predict_image
from src.run_training import build_image_list
from src.transforms import get_eval_transforms
from src.utils import create_project_directories, get_device


def save_error_examples(
    test_paths,
    metrics,
    model,
    device,
):
    """Save false-positive and false-negative images and metadata."""

    false_positive_dir = ERROR_ANALYSIS_DIR / "false_positives"
    false_negative_dir = ERROR_ANALYSIS_DIR / "false_negatives"

    false_positive_dir.mkdir(parents=True, exist_ok=True)
    false_negative_dir.mkdir(parents=True, exist_ok=True)

    false_positive_indices, false_negative_indices = get_error_indices(
        labels=metrics["labels"],
        predictions=metrics["predictions"],
    )

    rows = []

    error_groups = [
        (
            "false_positive",
            false_positive_indices,
            false_positive_dir,
            0,
            1,
        ),
        (
            "false_negative",
            false_negative_indices,
            false_negative_dir,
            1,
            0,
        ),
    ]

    for (
        error_type,
        indices,
        output_dir,
        true_label,
        predicted_label,
    ) in error_groups:

        for error_number, index in enumerate(indices, start=1):
            source_path = Path(test_paths[index])

            image = Image.open(source_path).convert("RGB")

            prediction = predict_image(
                model=model,
                image=image,
                device=device,
            )

            output_name = (
                f"{error_number:03d}_{source_path.name}"
            )

            output_path = output_dir / output_name

            shutil.copy2(
                source_path,
                output_path,
            )

            rows.append(
                {
                    "error_type": error_type,
                    "filename": source_path.name,
                    "true_class": CLASS_NAMES[true_label],
                    "predicted_class": CLASS_NAMES[predicted_label],
                    "confidence": prediction["confidence"],
                    "source_path": str(source_path),
                    "saved_path": str(output_path),
                }
            )

    metadata_path = METRICS_DIR / "error_analysis.csv"

    with metadata_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "error_type",
            "filename",
            "true_class",
            "predicted_class",
            "confidence",
            "source_path",
            "saved_path",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    return (
        false_positive_indices,
        false_negative_indices,
        metadata_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the trained defect classifier."
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
        "--checkpoint",
        type=Path,
        default=BEST_MODEL_PATH,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
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

    args = parser.parse_args()

    create_project_directories()

    image_paths, labels = build_image_list(
        normal_dir=args.normal_dir,
        defective_dir=args.defective_dir,
    )

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

    print("=== DATASET ===")
    print(f"Total images: {len(image_paths)}")
    print(f"Train: {len(train_paths)}")
    print(f"Validation: {len(val_paths)}")
    print(f"Test: {len(test_paths)}")

    print("\nTest class counts:")
    print(f"Normal: {test_labels.count(0)}")
    print(f"Defective: {test_labels.count(1)}")

    test_dataset = ProductDefectDataset(
        image_paths=test_paths,
        labels=test_labels,
        transform=get_eval_transforms(),
    )

    device = get_device()

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"\nDevice: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    confusion_matrix_path = (
        PLOTS_DIR / "confusion_matrix.png"
    )

    save_confusion_matrix(
        matrix=metrics["confusion_matrix"],
        output_path=confusion_matrix_path,
        class_names=CLASS_NAMES,
    )

    (
        false_positive_indices,
        false_negative_indices,
        error_metadata_path,
    ) = save_error_examples(
        test_paths=test_paths,
        metrics=metrics,
        model=model,
        device=device,
    )

    metrics_to_save = {
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1_score": float(metrics["f1"]),
        "confusion_matrix": metrics[
            "confusion_matrix"
        ].tolist(),
        "test_samples": len(test_paths),
        "normal_test_samples": test_labels.count(0),
        "defective_test_samples": test_labels.count(1),
        "false_positives": len(false_positive_indices),
        "false_negatives": len(false_negative_indices),
    }

    metrics_path = METRICS_DIR / "test_metrics.json"

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics_to_save,
            file,
            indent=2,
        )

    print("\n=== TEST METRICS ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    print("\n=== ERROR ANALYSIS ===")
    print(
        f"False positives: {len(false_positive_indices)}"
    )
    print(
        f"False negatives: {len(false_negative_indices)}"
    )

    print("\nArtifacts saved:")
    print(f"Metrics: {metrics_path}")
    print(f"Confusion matrix: {confusion_matrix_path}")
    print(f"Error metadata: {error_metadata_path}")
    print(
        "False positives:",
        ERROR_ANALYSIS_DIR / "false_positives",
    )
    print(
        "False negatives:",
        ERROR_ANALYSIS_DIR / "false_negatives",
    )


if __name__ == "__main__":
    main()
