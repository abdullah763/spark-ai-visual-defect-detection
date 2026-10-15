from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate a model and return classification metrics."""

    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu()

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.tolist())

    metrics = {
        "accuracy": accuracy_score(
            all_labels,
            all_predictions,
        ),
        "precision": precision_score(
            all_labels,
            all_predictions,
            pos_label=1,
            zero_division=0,
        ),
        "recall": recall_score(
            all_labels,
            all_predictions,
            pos_label=1,
            zero_division=0,
        ),
        "f1": f1_score(
            all_labels,
            all_predictions,
            pos_label=1,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            all_labels,
            all_predictions,
        ),
        "labels": all_labels,
        "predictions": all_predictions,
    }

    return metrics


def save_confusion_matrix(
    matrix,
    output_path: Path,
    class_names: list[str],
) -> None:
    """Save a confusion matrix figure."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots()

    image = ax.imshow(matrix)

    ax.figure.colorbar(
        image,
        ax=ax,
    )

    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
            )

    fig.tight_layout()
    fig.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.close(fig)


def get_error_indices(
    labels: list[int],
    predictions: list[int],
) -> tuple[list[int], list[int]]:
    """Return false-positive and false-negative sample indices."""

    false_positives = []
    false_negatives = []

    for index, (label, prediction) in enumerate(
        zip(labels, predictions)
    ):
        if label == 0 and prediction == 1:
            false_positives.append(index)

        elif label == 1 and prediction == 0:
            false_negatives.append(index)

    return false_positives, false_negatives
