"""
Data processing for the image classification pipeline.
Used by the notebook, the API (prediction + retraining) and the UI.
"""
import io
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test"
UPLOAD_DIR = ROOT / "data" / "uploads"

IMG_SIZE = (224, 224)   
BATCH_SIZE = 32
SEED = 42
IMG_EXTS = {".png", ".jpg", ".jpeg"}

AUTOTUNE = tf.data.AUTOTUNE


def clean_dataset(root: Path) -> int:
    """Delete unreadable/corrupt images so tf.data never crashes mid-epoch.
    Returns the number of files removed."""
    removed = 0
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                p.unlink()
                removed += 1
    return removed


def get_class_names(train_dir: Path = TRAIN_DIR) -> list:
    """Sorted class names. Sorting fixes label order (0, 1, ...) so the
    index the model outputs always maps to the same class."""
    return sorted(d.name for d in Path(train_dir).iterdir() if d.is_dir())


def count_images(root: Path) -> dict:
    """{class_name: number_of_images} - used for visualisations."""
    root = Path(root)
    if not root.exists():
        return {}
    return {
        d.name: sum(1 for f in d.iterdir() if f.suffix.lower() in IMG_EXTS)
        for d in sorted(root.iterdir()) if d.is_dir()
    }


def get_augmentation() -> tf.keras.Sequential:
    """Mild augmentation. No vertical flips: chest X-rays always have the
    same orientation, so flipping them upside down is unrealistic."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def load_train_val(train_dir: Path = TRAIN_DIR, val_split: float = 0.2,
                   class_names: list | None = None, augment: bool = True):
    """Returns (train_ds, val_ds). Pixels stay 0-255: the model rescales
    them internally, so training and serving cannot disagree."""
    class_names = class_names or get_class_names(train_dir)
    common = dict(
        directory=str(train_dir),
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=val_split,
        seed=SEED,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(subset="training", **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(subset="validation", **common)

    if augment:
        aug = get_augmentation()
        train_ds = train_ds.map(lambda x, y: (aug(x, training=True), y),
                                num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    return train_ds, val_ds


def load_test(test_dir: Path = TEST_DIR, class_names: list | None = None):
    """Test set: no augmentation, no shuffling (keeps labels aligned with
    predictions when we compute metrics)."""
    class_names = class_names or get_class_names(TRAIN_DIR)
    ds = tf.keras.utils.image_dataset_from_directory(
        str(test_dir),
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return ds.prefetch(AUTOTUNE)


def preprocess_image_bytes(data: bytes) -> np.ndarray:
    """Raw uploaded bytes -> float32 array of shape (1, H, W, 3), 0-255."""
    img = Image.open(io.BytesIO(data)).convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(np.asarray(img, dtype="float32"), axis=0)


def preprocess_image_path(path) -> np.ndarray:
    with open(path, "rb") as f:
        return preprocess_image_bytes(f.read())