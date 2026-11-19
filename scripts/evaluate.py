"""Evaluate the deployed model on the labelled State Farm images and write models/evaluation.json.

The Kaggle dataset needs a Kaggle account and accepting the competition rules, so it is not in this repository.
Download it once:
    kaggle competitions download -c state-farm-distracted-driver-detection
    unzip state-farm-distracted-driver-detection.zip -d data/raw

Then (from the repository root):
    python scripts/evaluate.py --data data/raw                   # the notebook's validation subset (4,481 images)
    python scripts/evaluate.py --data data/raw --subset all      # all 22,424 labelled images

"validation" rebuilds the subset Keras held out in the notebook: validation_split=0.2 takes the first 20% of each
class folder's sorted file list. The State Farm file names were kept, so the subset is the same files, assuming the
published weights came from that run. The web app's Evaluation page shows the result once the file exists.
"""
import argparse
import csv
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))
from detector.tfjs_loader import IMAGE_SIZE, build_model  # noqa: E402

VALIDATION_SPLIT = 0.2
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_images(train_dir, codes, subset):
    items = []
    for idx, code in enumerate(codes):
        files = sorted(p for p in (train_dir / code).iterdir() if p.suffix.lower() in EXTENSIONS)
        if subset == "validation":
            files = files[:int(VALIDATION_SPLIT * len(files))]
        items += [(p, idx) for p in files]
    return items


def load_batch(paths):
    return np.stack([np.asarray(Image.open(p).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR),
                                np.float32) for p in paths])


def per_class_report(cm, codes):
    report = []
    for i, code in enumerate(codes):
        tp = cm[i, i]
        precision = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        recall = tp / cm[i].sum() if cm[i].sum() else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report.append({"code": code, "precision": round(float(precision), 4), "recall": round(float(recall), 4),
                       "f1": round(float(f1), 4), "support": int(cm[i].sum())})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="folder with imgs/train/c0..c9 and driver_imgs_list.csv")
    parser.add_argument("--subset", choices=("validation", "all"), default="validation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "evaluation.json")
    args = parser.parse_args()

    metrics = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
    codes = [c["code"] for c in metrics["classes"]]
    safe = codes.index("c0")
    train_dir = args.data / "imgs" / "train"
    if not train_dir.is_dir():
        train_dir = args.data / "train"
    items = list_images(train_dir, codes, args.subset)
    if not items:
        raise SystemExit(f"No images found under {train_dir}")

    drivers = {}
    driver_csv = args.data / "driver_imgs_list.csv"
    if driver_csv.exists():
        with driver_csv.open(newline="") as f:
            drivers = {row["img"]: row["subject"] for row in csv.DictReader(f)}

    model = build_model(ROOT / "models" / "tfjs")
    preds = []
    t0 = time.time()
    for start in range(0, len(items), args.batch_size):
        batch = items[start:start + args.batch_size]
        preds.append(model.predict(load_batch([p for p, _ in batch]), verbose=0).argmax(1))
        print(f"\r{min(start + args.batch_size, len(items)):,}/{len(items):,} images", end="", flush=True)
    print(f"  ({time.time() - t0:.0f}s)")
    y_pred = np.concatenate(preds)
    y_true = np.array([label for _, label in items])

    n = len(codes)
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)

    # Business view: distracted (any of c1-c9) versus safe driving (c0)
    true_distracted, pred_distracted = y_true != safe, y_pred != safe
    binary = {
        "distracted_recall": round(float((pred_distracted & true_distracted).sum() / true_distracted.sum()), 4),
        "false_alarm_rate": round(float((pred_distracted & ~true_distracted).sum() / max((~true_distracted).sum(), 1)), 4),
        "accuracy": round(float((pred_distracted == true_distracted).mean()), 4),
    }
    by_driver = {}
    for (path, _), t, p in zip(items, y_true, y_pred):
        subject = drivers.get(path.name)
        if subject:
            d = by_driver.setdefault(subject, [0, 0])
            d[0] += int(t == p)
            d[1] += 1

    result = {
        "generated": date.today().isoformat(), "subset": args.subset, "n_images": len(items),
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "confusion_matrix": cm.tolist(), "per_class": per_class_report(cm, codes), "binary": binary,
        "per_driver": [{"driver": k, "accuracy": round(c / t, 4), "n": t} for k, (c, t) in sorted(by_driver.items())],
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Accuracy {result['accuracy']:.4f} on {len(items):,} images; "
          f"distracted recall {binary['distracted_recall']:.4f}, false alarms {binary['false_alarm_rate']:.4f}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
