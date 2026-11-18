"""
Data acquisition: get the Chest X-Ray Pneumonia dataset and build
data/train and data/test as <class_name>/<image files>.

Option A (automatic):  python scripts/download_data.py
Option B (manual):     download the zip from Kaggle, unzip it into
                       data/raw/ (so data/raw/chest_xray/train exists),
                       then run:  python scripts/download_data.py

Run from the project root.
"""
import random
import shutil
from pathlib import Path

from PIL import Image

# ---- Config ---------------------------------------------------------------
KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"
MAX_TRAIN_PER_CLASS = 2000   # softens class imbalance + keeps repo small
SAVE_SIZE = (256, 256)       # stored size; the model input is 224x224
TEST_FRACTION = 0.2          # only used if the dataset has no train/test split
SEED = 42
IMG_EXTS = {".png", ".jpg", ".jpeg"}

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test"


def list_images(folder: Path) -> list:
    """Image files in a folder, skipping hidden/macOS junk files."""
    if not folder.is_dir():
        return []
    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith(".")
    ]


def find_dataset_root(base: Path):
    """Find the folder that directly contains both train/ and test/.
    (The Kaggle zip nests chest_xray/chest_xray and adds __MACOSX.)"""
    for d in [base, *base.rglob("*")]:
        if (d.is_dir() and "__MACOSX" not in d.parts
                and (d / "train").is_dir() and (d / "test").is_dir()):
            return d
    return None


def get_source() -> Path:
    if RAW_DIR.exists() and any(RAW_DIR.rglob("*.jpeg")):
        print("Using manually downloaded data in", RAW_DIR)
        return RAW_DIR
    import kagglehub  # imported here so manual mode works without Kaggle login
    print("Downloading dataset via kagglehub ...")
    path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print("Downloaded to:", path)
    return path


def save_resized(src: Path, dst: Path):
    """Grayscale X-rays -> RGB (MobileNetV2 expects 3 channels), resized JPEG."""
    with Image.open(src) as im:
        im.convert("RGB").resize(SAVE_SIZE).save(dst, "JPEG", quality=90)


def write_split(files_by_class: dict, out_root: Path, cap=None, rng=None):
    for name, files in files_by_class.items():
        files = sorted(files)
        if cap:
            rng.shuffle(files)
            files = files[:cap]
        out = out_root / name
        out.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(files):
            save_resized(f, out / f"{name}_{i}.jpg")
        print(f"  {out_root.name}/{name}: {len(files)} images")


def main():
    rng = random.Random(SEED)
    base = get_source()
    root = find_dataset_root(base)
    if root is None:
        raise SystemExit("Could not find train/ and test/ folders in the dataset.")
    print("Dataset root:", root)

    for d in (TRAIN_DIR, TEST_DIR):
        if d.exists():
            shutil.rmtree(d)

    class_names = sorted(d.name for d in (root / "train").iterdir() if d.is_dir())
    train_files, test_files = {}, {}
    for cls in class_names:
        label = cls.title()  # NORMAL -> Normal, PNEUMONIA -> Pneumonia
        # merge the tiny original val set (16 images) into train
        train_files[label] = list_images(root / "train" / cls) + list_images(root / "val" / cls)
        test_files[label] = list_images(root / "test" / cls)

    write_split(train_files, TRAIN_DIR, cap=MAX_TRAIN_PER_CLASS, rng=rng)
    write_split(test_files, TEST_DIR)
    print("Done. Classes:", sorted(train_files))


if __name__ == "__main__":
    main()