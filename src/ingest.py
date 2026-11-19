"""
Ingestion of user-uploaded training images: validate -> preprocess -> save -> record in DB.
"""
import hashlib
import io
import uuid
from pathlib import Path

from PIL import Image

from src import database as db
from src.preprocessing import ROOT, UPLOAD_DIR

SAVE_SIZE = (256, 256)   # same stored size as the original training data


def ingest_image(data: bytes, label: str, original_name: str = "") -> str:
    """Returns 'saved', 'duplicate' or 'invalid'.

    Preprocessing applied to every upload (same as the training data):
      1. verify it is a real, readable image
      2. convert to RGB (X-rays are often grayscale)
      3. resize to 256x256 and store as JPEG
      4. skip exact duplicates (SHA-1 of the original bytes)
    """
    try:
        Image.open(io.BytesIO(data)).verify()
        img = Image.open(io.BytesIO(data)).convert("RGB").resize(SAVE_SIZE)
    except Exception:
        return "invalid"

    digest = hashlib.sha1(data).hexdigest()
    if db.has_hash(digest):
        return "duplicate"

    out_dir = UPLOAD_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{uuid.uuid4().hex}.jpg"
    img.save(path, "JPEG", quality=90)

    if not db.add_upload(original_name, label, str(path.relative_to(ROOT)), digest):
        path.unlink(missing_ok=True)   # lost a race with an identical upload
        return "duplicate"
    return "saved"