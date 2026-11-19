"""
FastAPI service: prediction, uptime, bulk upload, retraining (manual + automatic trigger),
metrics and data insights.

Run from the project root:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
Docs (Swagger):  http://localhost:8000/docs
"""
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src import database as db
from src import insights as insights_mod
from src import model as M
from src import prediction as P
from src.ingest import ingest_image
from src.preprocessing import TEST_DIR, TRAIN_DIR, UPLOAD_DIR, count_images, get_class_names
from src.retrain import run_retraining

START_TIME = time.time()
START_ISO = datetime.now(timezone.utc).isoformat(timespec="seconds")

AUTO_RETRAIN = os.getenv("AUTO_RETRAIN", "true").lower() == "true"
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", 50))   # pending uploads that trigger retraining
MAX_FILE_MB = 10

_stats = {"predictions": 0, "errors": 0, "total_latency_ms": 0.0}
_stats_lock = threading.Lock()

_retrain = {"running": False, "trigger": None, "started_at": None,
            "last_result": None, "last_error": None}
_retrain_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    try:
        P.load_model()          # warm up so the first request is fast
    except FileNotFoundError:
        pass                    # API still starts; /predict answers 503 until a model exists
    yield


app = FastAPI(title="Chest X-ray Pneumonia MLOps API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ helpers
def _class_names() -> list:
    if M.CLASS_NAMES_PATH.exists():
        return json.loads(M.CLASS_NAMES_PATH.read_text())
    return get_class_names()


def _start_retrain(trigger: str) -> bool:
    """Start retraining in a background thread. False if one is already running."""
    with _retrain_lock:
        if _retrain["running"]:
            return False
        _retrain.update(running=True, trigger=trigger, last_error=None,
                        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def worker():
        try:
            result = run_retraining(trigger)
            if result["promoted"]:
                P.reload_model()                     # serve the new model immediately
            with _retrain_lock:
                _retrain.update(running=False, last_result=result, last_error=None)
        except Exception as exc:                     # noqa: BLE001
            with _retrain_lock:
                _retrain.update(running=False, last_error=str(exc))

    threading.Thread(target=worker, daemon=True).start()
    return True


# ------------------------------------------------------------------ endpoints
@app.get("/")
def root():
    return {"service": "pneumonia-mlops-api", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    """Model up-time and status (used by the UI)."""
    with _stats_lock:
        n, errs, total = _stats["predictions"], _stats["errors"], _stats["total_latency_ms"]
    with _retrain_lock:
        retraining = _retrain["running"]
    return {
        "status": "ok",
        "started_at": START_ISO,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "model_loaded": P.is_loaded(),
        "model_file_exists": M.MODEL_PATH.exists(),
        "model_version": P.model_version(),
        "retraining": retraining,
        "predictions_served": n,
        "prediction_errors": errs,
        "avg_latency_ms": round(total / n, 1) if n else None,
    }


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    data = file.file.read()
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {MAX_FILE_MB} MB)")
    t0 = time.perf_counter()
    try:
        result = P.predict_bytes(data)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    except Exception:                                # unreadable / not an image
        with _stats_lock:
            _stats["errors"] += 1
        raise HTTPException(400, "Could not read the file as an image (use .jpg/.jpeg/.png)")
    latency = (time.perf_counter() - t0) * 1000
    with _stats_lock:
        _stats["predictions"] += 1
        _stats["total_latency_ms"] += latency
    return {**result, "latency_ms": round(latency, 1), "model_version": P.model_version()}


@app.post("/upload")
async def upload(label: str = Form(...), files: list[UploadFile] = File(...)):
    """Bulk upload of training images for ONE class. Each image is validated,
    preprocessed, saved to data/uploads/<class>/ and recorded in the database."""
    names = {n.lower(): n for n in _class_names()}
    if label.lower() not in names:
        raise HTTPException(400, f"Unknown label '{label}'. Valid labels: {list(names.values())}")
    label = names[label.lower()]

    counts = {"saved": 0, "duplicate": 0, "invalid": 0}
    for f in files:
        data = await f.read()
        if len(data) > MAX_FILE_MB * 1024 * 1024:
            counts["invalid"] += 1
            continue
        counts[ingest_image(data, label, f.filename or "")] += 1

    pending = db.count_pending()
    triggered = False
    if AUTO_RETRAIN and pending >= RETRAIN_THRESHOLD:
        triggered = _start_retrain("auto-threshold")
    return {"label": label, **counts, "pending_for_retraining": pending,
            "auto_retrain_threshold": RETRAIN_THRESHOLD, "auto_retrain_started": triggered}


@app.post("/retrain")
def retrain():
    """Manual trigger (the UI's Retrain button)."""
    if sum(count_images(UPLOAD_DIR).values()) == 0:
        raise HTTPException(400, "No uploaded data yet. Upload images first.")
    if not _start_retrain("manual"):
        raise HTTPException(409, "Retraining is already running")
    return {"message": "Retraining started", "poll": "/retrain/status"}


@app.get("/retrain/status")
def retrain_status():
    with _retrain_lock:
        state = dict(_retrain)
    state["pending_uploads"] = db.count_pending()
    state["auto_retrain"] = AUTO_RETRAIN
    state["auto_retrain_threshold"] = RETRAIN_THRESHOLD
    return state


@app.get("/retrain/history")
def retrain_history(limit: int = 20):
    return db.list_runs(limit)


@app.get("/data/summary")
def data_summary():
    return {
        "train": count_images(TRAIN_DIR),
        "test": count_images(TEST_DIR),
        "uploads": count_images(UPLOAD_DIR),
        "pending_by_class": db.pending_by_class(),
        "pending_total": db.count_pending(),
    }


@app.get("/metrics")
def metrics():
    """Evaluation of the model currently in production (test set)."""
    if not M.METRICS_PATH.exists():
        raise HTTPException(404, "No metrics saved yet")
    return json.loads(M.METRICS_PATH.read_text())


@app.get("/insights")
def insights(refresh: bool = False):
    """Image-feature summaries per class for the dashboard (cached)."""
    try:
        return insights_mod.compute_insights(force=refresh)
    except FileNotFoundError:
        raise HTTPException(404, "Training data not found")