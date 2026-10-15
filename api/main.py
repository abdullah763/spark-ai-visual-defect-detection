import logging
import time
from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from src.config import BEST_MODEL_PATH
from src.predict import load_model, predict_image
from src.utils import get_device


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

device = get_device()
model = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the API starts."""

    global model

    logger.info("Starting inference API")
    logger.info("Selected device: %s", device)

    if BEST_MODEL_PATH.exists():
        try:
            model = load_model(
                checkpoint_path=BEST_MODEL_PATH,
                device=device,
            )
            logger.info(
                "Model loaded from %s",
                BEST_MODEL_PATH,
            )
        except Exception:
            logger.exception("Failed to load model")
            model = None
    else:
        logger.warning(
            "Model checkpoint not found at %s",
            BEST_MODEL_PATH,
        )

    yield

    logger.info("Shutting down inference API")


app = FastAPI(
    title="Spark AI Visual Defect Detection API",
    description=(
        "Classifies product images as Normal or Defective."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health_check():
    """Return API and model availability."""

    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "device": str(device),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(file: UploadFile = File(...)):
    """Classify an uploaded product image."""

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available.",
        )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported file type. "
                "Use JPEG, PNG, or WEBP."
            ),
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Uploaded image exceeds the 10 MB limit.",
        )

    try:
        image = Image.open(BytesIO(file_bytes))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image.",
        )

    try:
        start_time = time.perf_counter()

        result = predict_image(
            model=model,
            image=image,
            device=device,
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.info(
            "Prediction=%s confidence=%.4f latency_ms=%.2f",
            result["predicted_class"],
            result["confidence"],
            latency_ms,
        )

        return result

    except Exception:
        logger.exception("Inference failed")

        raise HTTPException(
            status_code=500,
            detail="Inference failed.",
        )
