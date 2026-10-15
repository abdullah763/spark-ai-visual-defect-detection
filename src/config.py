from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Main directories
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Output subdirectories
METRICS_DIR = OUTPUT_DIR / "metrics"
PLOTS_DIR = OUTPUT_DIR / "plots"
ERROR_ANALYSIS_DIR = OUTPUT_DIR / "error_analysis"

# Model settings
MODEL_NAME = "resnet18"
NUM_CLASSES = 2
CLASS_NAMES = ["Normal", "Defective"]

# Image settings
IMAGE_SIZE = 224

# Training defaults
BATCH_SIZE = 8
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

# Model checkpoint
BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"
