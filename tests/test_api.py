from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

import api.main as api_module


def create_test_image() -> bytes:
    """Create an in-memory JPEG image for API testing."""

    image = Image.new(
        "RGB",
        (100, 100),
        "white",
    )

    buffer = BytesIO()
    image.save(buffer, format="JPEG")

    return buffer.getvalue()


def test_health_without_model():
    """Health endpoint should report degraded when no model is loaded."""

    with TestClient(api_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "degraded"
    assert data["model_loaded"] is False
    assert "device" in data


def test_predict_without_model():
    """Prediction should fail cleanly when no trained model exists."""

    with TestClient(api_module.app) as client:
        response = client.post(
            "/predict",
            files={
                "file": (
                    "test.jpg",
                    create_test_image(),
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 503


def test_unsupported_file_type(monkeypatch):
    """Reject unsupported upload types."""

    with TestClient(api_module.app) as client:
        monkeypatch.setattr(
            api_module,
            "model",
            object(),
        )

        response = client.post(
            "/predict",
            files={
                "file": (
                    "test.txt",
                    b"not an image",
                    "text/plain",
                )
            },
        )

    assert response.status_code == 415


def test_empty_image(monkeypatch):
    """Reject empty image uploads."""

    with TestClient(api_module.app) as client:
        monkeypatch.setattr(
            api_module,
            "model",
            object(),
        )

        response = client.post(
            "/predict",
            files={
                "file": (
                    "empty.jpg",
                    b"",
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 400


def test_invalid_image(monkeypatch):
    """Reject corrupt or invalid image data."""

    with TestClient(api_module.app) as client:
        monkeypatch.setattr(
            api_module,
            "model",
            object(),
        )

        response = client.post(
            "/predict",
            files={
                "file": (
                    "fake.jpg",
                    b"this is not image data",
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 400


def test_valid_prediction(monkeypatch):
    """Valid image should return class and confidence."""

    with TestClient(api_module.app) as client:
        monkeypatch.setattr(
            api_module,
            "model",
            object(),
        )

        monkeypatch.setattr(
            api_module,
            "predict_image",
            lambda model, image, device: {
                "predicted_class": "Normal",
                "confidence": 0.95,
            },
        )

        response = client.post(
            "/predict",
            files={
                "file": (
                    "test.jpg",
                    create_test_image(),
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_class"] == "Normal"
    assert data["confidence"] == 0.95
