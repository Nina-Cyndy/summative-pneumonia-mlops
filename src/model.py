"""
Model creation, training and evaluation for the chest X-ray classifier.

Design:
  * Transfer learning: MobileNetV2 pretrained on ImageNet (feature extractor)
  * Rescaling is a layer INSIDE the model, so training and serving can never
    disagree about preprocessing.
  * Two-phase training: (1) train the new head with the base frozen,
    (2) fine-tune the top of the base with a very small learning rate.
  * Regularisation: dropout, L2 weight decay, data augmentation, early stopping,
    learning-rate reduction on plateau, class weights for imbalance.
"""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.preprocessing import IMG_SIZE, TRAIN_DIR, count_images

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "pneumonia_model.h5"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
METRICS_PATH = MODEL_DIR / "metrics.json"


def build_baseline_cnn(num_classes: int = 2) -> tf.keras.Model:
    """Vanilla CNN with NO optimisation techniques (no pretrained weights, no
    dropout, no regularisation). Used only as a baseline to show that the
    optimised model is genuinely better."""
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="baseline_cnn")


def build_model(num_classes: int = 2, l2: float = 1e-4, dropout: float = 0.4,
                weights: str | None = "imagenet") -> tf.keras.Model:
    """MobileNetV2 transfer-learning model. Base starts FROZEN."""
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,), name="image")
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(inputs)

    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights=weights
    )
    base.trainable = False
    x = base(x, training=False)  

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(
        128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2)
    )(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return tf.keras.Model(inputs, outputs, name="pneumonia_mobilenetv2")


def get_base(model: tf.keras.Model) -> tf.keras.Model:
    """Return the nested pretrained base model (the one layer that is a Model)."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("No nested base model found")


def unfreeze_top(model: tf.keras.Model, n_layers: int = 30) -> None:
    """Unfreeze the last `n_layers` of the base for fine-tuning.
    BatchNorm layers stay frozen: updating their statistics with a small
    dataset and a tiny learning rate tends to hurt more than it helps."""
    base = get_base(model)
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def compile_model(model: tf.keras.Model, lr: float = 1e-3) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def get_callbacks(patience: int = 4) -> list:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1
        ),
    ]


def compute_class_weights(train_dir: Path = TRAIN_DIR) -> dict:
    """Weight = total / (n_classes * class_count): the rarer class counts more
    in the loss, so the model does not simply favour the majority class."""
    counts = count_images(train_dir)
    names = sorted(counts)
    total = sum(counts.values())
    return {i: total / (len(names) * counts[n]) for i, n in enumerate(names)}


def train_two_phase(model, train_ds, val_ds, class_weight=None,
                    epochs_head: int = 10, epochs_finetune: int = 10,
                    lr_head: float = 1e-3, lr_finetune: float = 1e-5,
                    unfreeze_layers: int = 30):
    """Phase 1: train the new head (base frozen). Phase 2: fine-tune the top
    of the base. Returns (history_head, history_finetune)."""
    compile_model(model, lr_head)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=epochs_head,
                   class_weight=class_weight, callbacks=get_callbacks())

    unfreeze_top(model, unfreeze_layers)
    compile_model(model, lr_finetune)  # must recompile after changing trainable
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=epochs_finetune,
                   class_weight=class_weight, callbacks=get_callbacks())
    return h1, h2


def evaluate_model(model, test_ds, class_names: list, positive_class: str = "Pneumonia"):
    """Evaluate on a dataset (shuffle=False!). Returns a metrics dict plus
    y_true / y_prob for plotting."""
    pos = class_names.index(positive_class)
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_prob = model.predict(test_ds, verbose=0)
    y_pred = y_prob.argmax(axis=1)

    loss = float(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_prob).numpy().mean())
    y_bin = (y_true == pos).astype(int)

    metrics = {
        "loss": loss,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_bin, (y_pred == pos).astype(int), zero_division=0)),
        "recall": float(recall_score(y_bin, (y_pred == pos).astype(int), zero_division=0)),
        "f1": float(f1_score(y_bin, (y_pred == pos).astype(int), zero_division=0)),
        "auc": float(roc_auc_score(y_bin, y_prob[:, pos])),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "class_names": class_names,
        "positive_class": positive_class,
        "n_samples": int(len(y_true)),
    }
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    return metrics, report, y_true, y_prob


def save_artifacts(model, class_names: list, metrics: dict | None = None,
                   model_path: Path = MODEL_PATH) -> None:
    """Save the model (.h5), the class-name order and the latest metrics.
    The API and UI read these files."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    CLASS_NAMES_PATH.write_text(json.dumps(class_names))
    if metrics is not None:
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))


def load_saved_model(model_path: Path = MODEL_PATH) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)