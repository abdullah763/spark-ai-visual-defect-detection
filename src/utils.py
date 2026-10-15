import random

import numpy as np
import torch

from src.config import (
    MODEL_DIR,
    METRICS_DIR,
    PLOTS_DIR,
    ERROR_ANALYSIS_DIR,
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Select CUDA, Apple MPS, or CPU depending on availability."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_project_directories() -> None:
    """Create directories required for models, metrics, plots, and errors."""
    directories = [
        MODEL_DIR,
        METRICS_DIR,
        PLOTS_DIR,
        ERROR_ANALYSIS_DIR / "false_positives",
        ERROR_ANALYSIS_DIR / "false_negatives",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
