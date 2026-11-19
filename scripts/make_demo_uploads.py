"""
Create a small folder of images to practise the upload -> retrain demo with.

    python -m scripts.make_demo_uploads

Writes demo_uploads/Normal and demo_uploads/Pneumonia (30 images each), taken from
the ORIGINAL Kaggle TRAIN folder, never from data/test. (Test images must stay unseen,
otherwise the evaluation that decides whether a retrained model is promoted is contaminated.)
Some of these images may already be in data/train; that is fine for demonstrating the
mechanics of upload -> preprocessing -> retraining.
"""
import random
import shutil
from pathlib import Path

from scripts.download_data import find_dataset_root, get_source, list_images

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo_uploads"
N = 30


def main():
    root = find_dataset_root(get_source())
    if root is None:
        raise SystemExit("Original dataset not found.")
    rng = random.Random(7)
    if OUT.exists():
        shutil.rmtree(OUT)
    for cls in sorted(d.name for d in (root / "train").iterdir() if d.is_dir()):
        files = list_images(root / "train" / cls)
        rng.shuffle(files)
        out = OUT / cls.title()
        out.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(files[:N]):
            shutil.copyfile(f, out / f"demo_{i}{f.suffix.lower()}")
        print(f"{out.name}: {min(N, len(files))} images")
    print("Done ->", OUT)


if __name__ == "__main__":
    main()