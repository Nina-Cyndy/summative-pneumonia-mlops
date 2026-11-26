"""
Prediction: load the saved model once, then classify uploaded images.
Used by the notebook and the FastAPI service.

Production behaviour:
  * the model file is watched: when a retrain is promoted (file replaced), every serving
    process reloads it automatically on its next request
  * inference runs through a compiled tf.function (faster than calling the model eagerly)
  * TF_INTRA_THREADS / TF_INTER_THREADS cap TensorFlow's CPU threads so that several
    containers can share one machine fairly
"""
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.preprocessing import IMG_SIZE, get_class_names, preprocess_image_bytes, preprocess_image_path

for _env, _setter in (("TF_INTRA_THREADS", tf.config.threading.set_intra_op_parallelism_threads),
                      ("TF_INTER_THREADS", tf.config.threading.set_inter_op_parallelism_threads)):
    _n = int(os.getenv(_env, "0"))
    if _n > 0:
        try:
            _setter(_n)
        except RuntimeError:        
            pass

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "pneumonia_model.h5"
CLASS_NAMES_PATH = ROOT / "models" / "class_names.json"

_model = None
_class_names = None
_infer = None
_loaded_mtime = None
_lock = threading.Lock()


def _build_infer(model):
    """Compiled inference function; falls back to eager calls if tracing fails."""
    try:
        fn = tf.function(lambda x: model(x, training=False),
                         input_signature=[tf.TensorSpec([None, *IMG_SIZE, 3], tf.float32)])
        fn(np.zeros((1, *IMG_SIZE, 3), dtype="float32"))    
        return fn
    except Exception:                                         
        return lambda x: model(x, training=False)


def load_model(force: bool = False):
    """Return (model, class_names). (Re)loads when forced, on first use, or when the
    model file on disk has changed since it was loaded."""
    global _model, _class_names, _infer, _loaded_mtime
    with _lock:
        mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
        if _model is None or force or (mtime is not None and mtime != _loaded_mtime):
            if mtime is None:
                raise FileNotFoundError(f"No trained model at {MODEL_PATH}. Train one first.")
            _model = tf.keras.models.load_model(MODEL_PATH, compile=False)   
            _class_names = (json.loads(CLASS_NAMES_PATH.read_text())
                            if CLASS_NAMES_PATH.exists() else get_class_names())
            _infer = _build_infer(_model)
            _loaded_mtime = mtime
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
    _, class_names = load_model()
    probs = _infer(batch).numpy()[0]
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