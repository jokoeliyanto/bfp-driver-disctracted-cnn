"""Build models/metrics.json, the web app's sample images and the evaluation figures.

Everything comes from files in this repository, so the script runs without the Kaggle dataset:

1. Training history: parsed from the Keras logs saved in notebooks/original_bangkit_capstone.ipynb
   (baseline CNN from scratch, and MobileNet transfer learning).
2. The deployed model (models/tfjs): parameter count, file size and CPU latency, measured here.
3. A preprocessing check: the eight example frames in images/imageexample.png are classified with raw [0, 255]
   pixels and with [0, 1] pixels (the notebook's rescale=1/255), to confirm which one the weights expect.

Usage (from the repository root):
    python scripts/build_metrics.py
"""
import json
import platform
import re
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))
from detector.tfjs_loader import IMAGE_SIZE, build_model  # noqa: E402

NOTEBOOK = ROOT / "notebooks" / "original_bangkit_capstone.ipynb"
MODEL_DIR = ROOT / "models" / "tfjs"
SAMPLES_DIR = ROOT / "webapp" / "detector" / "static" / "detector" / "samples"
FIGURES = ROOT / "images"

# Kaggle class codes, in the output order of the published model (verified in check_preprocessing)
CLASSES = [
    ("c0", "Safe driving", False),
    ("c1", "Texting (right hand)", True),
    ("c2", "Talking on the phone (right)", True),
    ("c3", "Texting (left hand)", True),
    ("c4", "Talking on the phone (left)", True),
    ("c5", "Operating the radio", True),
    ("c6", "Drinking", True),
    ("c7", "Reaching behind", True),
    ("c8", "Hair and makeup", True),
    ("c9", "Talking to passenger", True),
]
# Tile boxes of the 4 x 2 grid in images/imageexample.png (left, top, right, bottom)
TILES = [(x0 + 1, y0 + 1, x1 - 1, y1 - 1)
         for y0, y1 in ((8, 158), (181, 332)) for x0, x1 in ((8, 207), (231, 430), (454, 654), (678, 877))]
TILE_NAMES = ["img_1", "img_2", "img_3", "img_4", "img_7", "img_8", "img_9", "img_10"]

EPOCH = re.compile(r"(\d+)/\d+ \[=+\] - (\d+)s (\d+)(m?s)/step - loss: ([\d.]+) - accuracy: ([\d.]+)"
                   r" - val_loss: ([\d.]+) - val_accuracy: ([\d.]+)")
FOUND = re.compile(r"Found (\d+) images belonging to (\d+) classes")
PARAMS = re.compile(r"(Total|Trainable|Non-trainable) params: ([\d,]+)")


def cell_text(cell):
    return "".join("".join(o.get("text", "")) for o in cell.get("outputs", []))


def parse_history(text):
    epochs = []
    for steps, secs, step_time, unit, loss, acc, val_loss, val_acc in EPOCH.findall(text):
        epochs.append({"epoch": len(epochs) + 1, "loss": float(loss), "accuracy": float(acc),
                       "val_loss": float(val_loss), "val_accuracy": float(val_acc),
                       "seconds": int(secs), "steps": int(steps)})
    return epochs


def summarise_run(key, name, description, history, params, stopped_early):
    best = min(history, key=lambda e: e["val_loss"])     # the checkpoint kept (save_best_only on val_loss)
    return {
        "key": key, "name": name, "description": description, "history": history,
        "params": params, "epochs_run": len(history), "stopped_early": stopped_early,
        "best_epoch": best["epoch"], "best_val_accuracy": best["val_accuracy"], "best_val_loss": best["val_loss"],
        "best_train_accuracy": best["accuracy"],
        "seconds_per_epoch": round(statistics.mean(e["seconds"] for e in history)),
        "total_minutes": round(sum(e["seconds"] for e in history) / 60, 1),
    }


def parse_notebook():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = nb["cells"]
    found = [tuple(map(int, m)) for m in FOUND.findall(cell_text(cells[5]))]
    params = {}
    for idx, key in ((8, "baseline"), (14, "transfer")):
        params[key] = {k.lower().replace("-", "_"): int(v.replace(",", "")) for k, v in PARAMS.findall(cell_text(cells[idx]))}
    baseline_log, transfer_log = cell_text(cells[9]), cell_text(cells[15])
    runs = [
        summarise_run("baseline", "Baseline CNN (from scratch)",
                      "4 Conv2D + MaxPooling blocks (512-256-128-64 filters), Dense 512, Dense 10. Adam 1e-4.",
                      parse_history(baseline_log), params["baseline"], "early stopping" in baseline_log),
        summarise_run("transfer", "MobileNet transfer learning (deployed)",
                      "Frozen ImageNet MobileNet + Conv2D 1024 + Dropout 0.2 + GAP + Dense 10. Adam.",
                      parse_history(transfer_log), params["transfer"], "early stopping" in transfer_log),
    ]
    return {"train_images": found[0][0], "val_images": found[1][0], "n_classes": found[0][1]}, runs


def crop_samples():
    grid = Image.open(FIGURES / "imageexample.png").convert("RGB")
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    samples = []
    for i, (box, name) in enumerate(zip(TILES, TILE_NAMES), 1):
        tile = grid.crop(box)
        filename = f"sample_{i:02d}.png"
        tile.save(SAMPLES_DIR / filename, optimize=True)
        samples.append({"file": filename, "source": name, "image": tile})
    return samples


def to_input(image, scale):
    return np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR), np.float32) * scale


def check_preprocessing(model, samples):
    """Predict the example frames with raw pixels and with the notebook's 1/255 rescaling."""
    out = {}
    for key, scale in (("raw_0_255", 1.0), ("rescaled_0_1", 1 / 255)):
        proba = model.predict(np.stack([to_input(s["image"], scale) for s in samples]), verbose=0)
        top = proba.argmax(1)
        out[key] = {"predicted": [CLASSES[i][0] for i in top],
                    "distinct_classes": int(len(set(top))),
                    "mean_top_probability": round(float(proba.max(1).mean()), 3),
                    "spread": round(float(proba.std(0).mean()), 4)}
    return out


def benchmark(model, image, runs=40):
    import tensorflow as tf

    x = tf.constant(to_input(image, 1.0)[None])
    for _ in range(5):
        model(x, training=False)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x, training=False)
        times.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": round(statistics.median(times), 1), "p90_ms": round(float(np.percentile(times, 90)), 1),
            "runs": runs, "batch_size": 1, "device": f"CPU ({platform.processor() or platform.machine()})",
            "tensorflow": tf.__version__}


def plot_curves(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401  (MaxNLocator)

    colors = {"baseline": "#EA4335", "transfer": "#4285F4"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=130)
    for ax, metric, title in ((axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")):
        for run in runs:
            ep = [e["epoch"] for e in run["history"]]
            c = colors[run["key"]]
            ax.plot(ep, [e[metric] for e in run["history"]], "--", color=c, alpha=.6, lw=1.6,
                    label=f"{run['name']} · train")
            ax.plot(ep, [e[f"val_{metric}"] for e in run["history"]], "-o", color=c, ms=3.5, lw=2,
                    label=f"{run['name']} · validation")
        ax.set_title(title, loc="left", fontweight="bold", color="#202124")
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid(alpha=.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "training_curves.png")
    plt.close(fig)


def main():
    dataset, runs = parse_notebook()
    samples = crop_samples()
    print("Building the model from models/tfjs ...")
    model = build_model(MODEL_DIR)
    size_bytes = sum(p.stat().st_size for p in MODEL_DIR.glob("group1-shard*.bin"))
    deployed = runs[1]
    if model.count_params() != deployed["params"]["total"]:
        raise SystemExit(f"Parameter count {model.count_params():,} differs from the notebook's "
                         f"{deployed['params']['total']:,}; models/tfjs is not the notebook model.")

    metrics = {
        "generated": date.today().isoformat(),
        "model_name": "MobileNet transfer learning",
        "classes": [{"code": c, "name": n, "distracted": d} for c, n, d in CLASSES],
        "image_size": IMAGE_SIZE,
        "preprocessing": "RGB, resized to 224 x 224, raw pixel values 0-255",
        "dataset": {
            "name": "State Farm Distracted Driver Detection (Kaggle)",
            "url": "https://www.kaggle.com/c/state-farm-distracted-driver-detection/data",
            "labelled_images": dataset["train_images"] + dataset["val_images"],
            "train_images": dataset["train_images"], "val_images": dataset["val_images"],
            "n_classes": dataset["n_classes"], "n_drivers": 26,
            "split": "Random 80/20 per class (Keras validation_split); the same drivers appear in both subsets",
        },
        "runs": runs,
        "deployed": {
            "run": "transfer", "format": "TensorFlow.js layers model (13 shards)",
            "params": model.count_params(), "size_mb": round(size_bytes / 1e6, 1),
            "cpu_latency": benchmark(model, samples[0]["image"]),
        },
        "preprocessing_check": check_preprocessing(model, samples),
        "samples": [{"file": s["file"], "source": s["source"]} for s in samples],
    }
    path = ROOT / "models" / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_curves(runs)
    print(f"Wrote {path.relative_to(ROOT)}, {len(samples)} samples and images/training_curves.png")
    print(json.dumps(metrics["preprocessing_check"], indent=2))
    print(metrics["deployed"])


if __name__ == "__main__":
    main()
