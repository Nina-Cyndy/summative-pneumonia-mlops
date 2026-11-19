"""
Prediction: load the saved model once, then classify uploaded images.
Used by the notebook and the FastAPI service.
"""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.preprocessing import get_class_names, preprocess_image_bytes, preprocess_image_path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "pneumonia_model.h5"
CLASS_NAMES_PATH = ROOT / "models" / "class_names.json"

_model = None
_class_names = None
_lock = threading.Lock()


def load_model(force: bool = False):
    """Load the model once and cache it. force=True reloads from disk
    (used after retraining so the API serves the new model)."""
    global _model, _class_names
    with _lock:
        if _model is None or force:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"No trained model at {MODEL_PATH}. Train one first.")
            _model = tf.keras.models.load_model(MODEL_PATH, compile=False)  # inference only
            if CLASS_NAMES_PATH.exists():
                _class_names = json.loads(CLASS_NAMES_PATH.read_text())
            else:
                _class_names = get_class_names()
    return _model, _class_names


def reload_model():
    return load_model(force=True)


def is_loaded() -> bool:
    return _model is not None


def model_version() -> str | None:
    """Last-modified time of the model file (changes whenever a retrain is promoted)."""
    if not MODEL_PATH.exists():
        return None
    return datetime.fromtimestamp(MODEL_PATH.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")


def _predict_array(batch: np.ndarray) -> dict:
    model, class_names = load_model()
    probs = model(batch, training=False).numpy()[0]   # faster than predict() for one image
    idx = int(np.argmax(probs))
    return {
        "label": class_names[idx],
        "confidence": float(probs[idx]),
        "probabilities": {name: float(p) for name, p in zip(class_names, probs)},
    }


def predict_bytes(data: bytes) -> dict:
    """Predict from raw uploaded image bytes."""
    return _predict_array(preprocess_image_bytes(data))


def predict_path(path) -> dict:
    """Predict from an image file path."""
    return _predict_array(preprocess_image_path(path))