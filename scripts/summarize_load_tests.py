"""
Turn Locust CSVs (results/n<N>_stats.csv) into a table and a chart for the README.

    python scripts/summarize_load_tests.py [results_dir]
"""
import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results"


def read_run(path: Path):
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["Name"] == "POST /predict":
                return {
                    "requests": int(row["Request Count"]),
                    "failures": int(row["Failure Count"]),
                    "rps": float(row["Requests/s"]),
                    "avg": float(row["Average Response Time"]),
                    "p50": float(row["50%"]),
                    "p95": float(row["95%"]),
                    "p99": float(row["99%"]),
                    "max": float(row["Max Response Time"]),
                }
    return None


runs = {}
for path in RESULTS.glob("n*_stats.csv"):
    m = re.match(r"n(\d+)_stats\.csv", path.name)
    run = read_run(path) if m else None
    if run:
        runs[int(m.group(1))] = run

if not runs:
    raise SystemExit(f"No Locust CSV files found in {RESULTS}. Run scripts/run_load_tests.sh first.")

counts = sorted(runs)
lines = ["| Inference containers | Requests | Failures | Throughput (req/s) | Avg (ms) | Median (ms) | p95 (ms) | p99 (ms) | Max (ms) |",
         "|---|---|---|---|---|---|---|---|---|"]
for n in counts:
    r = runs[n]
    lines.append(f"| {n} | {r['requests']:,} | {r['failures']} ({r['failures'] / max(r['requests'], 1):.1%}) | "
                 f"{r['rps']:.1f} | {r['avg']:.0f} | {r['p50']:.0f} | {r['p95']:.0f} | {r['p99']:.0f} | {r['max']:.0f} |")
table = "\n".join(lines)
(RESULTS / "summary.md").write_text(table + "\n")
print(table)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
x = [str(n) for n in counts]
axes[0].bar(x, [runs[n]["rps"] for n in counts], color="#4F46E5")
axes[0].set_title("Throughput (requests/second)")
axes[0].set_xlabel("Inference containers")
width = 0.27
pos = range(len(counts))
for i, (key, color) in enumerate((("p50", "#14B8A6"), ("p95", "#F59E0B"), ("p99", "#F0605D"))):
    axes[1].bar([p + (i - 1) * width for p in pos], [runs[n][key] for n in counts], width,
                label=key.replace("p50", "median"), color=color)
axes[1].set_xticks(list(pos))
axes[1].set_xticklabels(x)
axes[1].set_title("Response time (ms, lower is better)")
axes[1].set_xlabel("Inference containers")
axes[1].legend()
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(RESULTS / "load_test_summary.png", dpi=160)
print(f"\nSaved {RESULTS / 'summary.md'} and {RESULTS / 'load_test_summary.png'}")