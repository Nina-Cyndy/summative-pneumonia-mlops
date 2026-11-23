"""
Locust flood test for POST /predict.

Interactive UI:   locust -f locust/locustfile.py --host http://localhost:8000
Headless:         locust -f locust/locustfile.py --headless -u 50 -r 10 -t 60s \
                         --host http://localhost:8000 --csv results/n1

Images come from data/test (a fixed subset, so every run sends the same workload).
"""
import json
import os
import random
from collections import Counter
from pathlib import Path

from locust import HttpUser, between, events, task

ROOT = Path(__file__).resolve().parents[1]
IMAGES_PER_CLASS = 15


def _load_images():
    images = []
    for cls in ("Normal", "Pneumonia"):
        for p in sorted((ROOT / "data" / "test" / cls).glob("*.jpg"))[:IMAGES_PER_CLASS]:
            images.append((p.name, p.read_bytes()))
    if not images:
        raise SystemExit("No images found in data/test. Run scripts/download_data.py first.")
    return images


IMAGES = _load_images()
SERVED_BY = Counter()          # which container answered each request


class XrayUser(HttpUser):
    wait_time = between(0.05, 0.25)    # a "flood": very short think time

    @task
    def predict(self):
        name, data = random.choice(IMAGES)
        with self.client.post("/predict", files={"file": (name, data, "image/jpeg")},
                              name="POST /predict", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            try:
                body = resp.json()
            except ValueError:
                resp.failure("response is not JSON")
                return
            if body.get("label") not in ("Normal", "Pneumonia"):
                resp.failure("unexpected label")
                return
            SERVED_BY[body.get("served_by", "unknown")] += 1
            resp.success()


@events.quitting.add_listener
def report_distribution(environment, **kwargs):
    total = sum(SERVED_BY.values())
    if not total:
        return
    print("\nRequests answered per container:")
    for container, n in SERVED_BY.most_common():
        print(f"  {container}: {n} ({n / total:.0%})")
    out = os.getenv("SERVED_OUT")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(SERVED_BY, indent=2))