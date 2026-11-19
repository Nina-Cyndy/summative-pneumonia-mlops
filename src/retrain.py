"""
Retraining pipeline.

Flow:
  1. collect uploaded images (data/uploads/<class>/) recorded in the database
  2. mix them with a random "replay" sample of the original training data
     (so the model does not forget what it already knows)
  3. LOAD THE EXISTING SAVED MODEL (.h5) as the starting point (custom pre-trained model)
  4. fine-tune it (top of MobileNetV2 + head, small learning rate, early stopping)
  5. evaluate old vs new on the untouched test set
  6. promote the new model only if it is not worse (old one is archived)

Run from the API (button / automatic trigger) or from the command line:
    python -m src.retrain
"""
import json
import os
import random
import shutil
import time
from pathlib import Path

from src import database as db
from src import model as M
from src.preprocessing import (
    IMG_EXTS, ROOT, TRAIN_DIR, UPLOAD_DIR, count_images, load_test, load_train_val,
)

ARCHIVE_DIR = M.MODEL_DIR / "archive"
WORK_DIR = ROOT / "data" / "retrain_tmp"

REPLAY_PER_CLASS = int(os.getenv("REPLAY_PER_CLASS", 400))   # old images kept per class
EPOCHS = int(os.getenv("RETRAIN_EPOCHS", 5))
LEARNING_RATE = float(os.getenv("RETRAIN_LR", 3e-5))
UNFREEZE_LAYERS = int(os.getenv("RETRAIN_UNFREEZE", 30))
MIN_DELTA = float(os.getenv("RETRAIN_MIN_DELTA", 0.02))      # tolerated drop in F1 / recall
SEED = 42


def _images(folder: Path) -> list:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def build_training_folder(class_names: list) -> tuple:
    """Create data/retrain_tmp/<class>/ with all uploads + a replay sample of the
    original training data. Returns (folder, n_new, n_replay)."""
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    rng = random.Random(SEED)
    n_new = n_replay = 0
    for cls in class_names:
        out = WORK_DIR / cls
        out.mkdir(parents=True, exist_ok=True)

        for p in _images(UPLOAD_DIR / cls):
            shutil.copyfile(p, out / f"new_{p.name}")
            n_new += 1

        old = _images(TRAIN_DIR / cls)
        rng.shuffle(old)
        for p in old[:REPLAY_PER_CLASS]:
            shutil.copyfile(p, out / f"old_{p.name}")
            n_replay += 1
    return WORK_DIR, n_new, n_replay


def _load_class_names() -> list:
    if M.CLASS_NAMES_PATH.exists():
        return json.loads(M.CLASS_NAMES_PATH.read_text())
    return sorted(d.name for d in TRAIN_DIR.iterdir() if d.is_dir())


def _decide_promotion(old: dict, new: dict) -> tuple:
    ok_f1 = new["f1"] >= old["f1"] - MIN_DELTA
    ok_recall = new["recall"] >= old["recall"] - MIN_DELTA
    if ok_f1 and ok_recall:
        return True, (f"promoted: F1 {old['f1']:.3f} -> {new['f1']:.3f}, "
                      f"recall {old['recall']:.3f} -> {new['recall']:.3f}")
    return False, (f"rejected: new model is worse than the tolerance of {MIN_DELTA} "
                   f"(F1 {old['f1']:.3f} -> {new['f1']:.3f}, "
                   f"recall {old['recall']:.3f} -> {new['recall']:.3f}); old model kept")


def run_retraining(trigger: str = "manual") -> dict:
    """Retrain from the saved model. Returns a summary dict (also stored in the DB)."""
    db.init_db()
    run_id = db.start_run(trigger)
    started = time.time()
    try:
        if not M.MODEL_PATH.exists():
            raise FileNotFoundError("No saved model to retrain. Train one in the notebook first.")
        class_names = _load_class_names()
        if sum(count_images(UPLOAD_DIR).values()) == 0:
            raise ValueError("No uploaded data found. Upload images first.")

        up_to_id = db.max_upload_id()                 # uploads arriving later wait for next run
        folder, n_new, n_replay = build_training_folder(class_names)
        train_ds, val_ds = load_train_val(folder, val_split=0.2, class_names=class_names, augment=True)
        class_weight = M.compute_class_weights(folder)

        # --- the saved model is the starting point (custom pre-trained model)
        model = M.load_saved_model()
        M.unfreeze_top(model, UNFREEZE_LAYERS)
        M.compile_model(model, LEARNING_RATE)
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                            class_weight=class_weight, callbacks=M.get_callbacks(patience=2))
        epochs_run = len(history.history["loss"])

        # --- evaluate old vs new on the same untouched test set
        test_ds = load_test(class_names=class_names)
        new_metrics, _, _, _ = M.evaluate_model(model, test_ds, class_names)
        if M.METRICS_PATH.exists():
            old_metrics = json.loads(M.METRICS_PATH.read_text())
        else:
            old_model = M.load_saved_model()
            old_metrics, _, _, _ = M.evaluate_model(old_model, test_ds, class_names)

        promoted, reason = _decide_promotion(old_metrics, new_metrics)
        if promoted:
            ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            shutil.copy2(M.MODEL_PATH, ARCHIVE_DIR / f"pneumonia_model_{stamp}.h5")
            tmp_path = M.MODEL_DIR / "pneumonia_model.tmp.h5"
            M.save_artifacts(model, class_names, new_metrics, model_path=tmp_path)
            os.replace(tmp_path, M.MODEL_PATH)        # atomic swap: API never sees a half-written file

        db.mark_uploads_used(up_to_id)
        result = {
            "run_id": run_id, "trigger": trigger, "promoted": promoted, "message": reason,
            "n_new": n_new, "n_replay": n_replay, "epochs": epochs_run,
            "old_metrics": old_metrics, "new_metrics": new_metrics,
            "duration_seconds": round(time.time() - started, 1),
        }
        db.finish_run(run_id, "finished", n_new=n_new, n_replay=n_replay, epochs=epochs_run,
                      old_metrics=old_metrics, new_metrics=new_metrics,
                      promoted=int(promoted), message=reason)
        return result
    except Exception as exc:
        db.finish_run(run_id, "failed", message=str(exc))
        raise
    finally:
        shutil.rmtree(WORK_DIR, ignore_errors=True)


if __name__ == "__main__":
    print(json.dumps(run_retraining("cli"), indent=2))