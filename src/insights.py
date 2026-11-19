"""
Data insights for the UI: simple image features per class (computed on a sample, cached).
"""
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from src.preprocessing import IMG_EXTS, ROOT, TRAIN_DIR

CACHE_PATH = ROOT / "models" / "insights.json"


def _features(path: Path) -> dict:
    arr = np.asarray(Image.open(path).convert("L"), dtype="float32")
    h, w = arr.shape
    centre = arr[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    return {
        "brightness": float(arr.mean()),
        "contrast": float(arr.std()),
        "centre_brightness": float(centre.mean()),
    }


def compute_insights(sample_per_class: int = 150, force: bool = False) -> dict:
    if CACHE_PATH.exists() and not force:
        return json.loads(CACHE_PATH.read_text())

    rng = random.Random(42)
    rows = []
    for cls_dir in sorted(d for d in TRAIN_DIR.iterdir() if d.is_dir()):
        files = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        for p in rng.sample(files, min(sample_per_class, len(files))):
            rows.append({"class": cls_dir.name, **_features(p)})

    summary = {}
    for cls in sorted({r["class"] for r in rows}):
        sub = [r for r in rows if r["class"] == cls]
        summary[cls] = {k: float(np.mean([r[k] for r in sub]))
                        for k in ("brightness", "contrast", "centre_brightness")}
    result = {"sample_per_class": sample_per_class, "features": rows, "summary": summary}
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(result))
    return result