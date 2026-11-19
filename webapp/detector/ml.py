"""Inference for the distracted driver detector.

Loads the Bangkit team's published MobileNet model (models/tfjs, rebuilt in Keras by tfjs_loader) and applies the
preprocessing its weights expect: RGB, resized to 224 x 224, raw pixel values 0-255.

The model predicts one of the ten State Farm activities. The app turns that into the decision a driver alert needs:
any activity other than c0 (safe driving) counts as distracted.
"""
import base64
import io
import json
from functools import lru_cache

import numpy as np
from django.conf import settings
from PIL import Image, ImageOps, UnidentifiedImageError

from .tfjs_loader import IMAGE_SIZE, build_model

MAX_PIXELS = 40_000_000
PREVIEW_SIZE = (640, 640)
LOW_CONFIDENCE = 0.6
SAFE_CODE = "c0"


class InputError(ValueError):
    """Raised when an uploaded file cannot be classified."""


@lru_cache(maxsize=1)
def get_metrics():
    return json.loads(settings.METRICS_PATH.read_text(encoding="utf-8"))


def get_evaluation():
    """The full evaluation from scripts/evaluate.py, or None if it has not been run."""
    path = settings.EVALUATION_PATH
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@lru_cache(maxsize=1)
def get_model():
    return build_model(settings.MODEL_DIR)       # heavy (TensorFlow), deferred until the first prediction


def classes():
    return get_metrics()["classes"]


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def open_image(data):
    """Decode uploaded bytes into an RGB PIL image. Raises InputError on bad files."""
    if not data:
        raise InputError("The file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.width * im.height > MAX_PIXELS:
                raise InputError("The image is too large.")
            im = ImageOps.exif_transpose(im)        # respect phone camera orientation
            return im.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InputError("The file is not a readable image (use PNG, JPEG or WebP).") from exc


def to_model_input(image):
    return np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR), np.float32)


def preview_data_uri(image):
    """A small JPEG of the image, embedded in the page so uploads are never stored."""
    thumb = image.copy()
    thumb.thumbnail(PREVIEW_SIZE)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def _format(proba):
    info = classes()
    ranked = sorted(({**c, "probability": float(p)} for c, p in zip(info, proba)),
                    key=lambda c: c["probability"], reverse=True)
    best = ranked[0]
    p_safe = next(c["probability"] for c in ranked if c["code"] == SAFE_CODE)
    return {
        "code": best["code"], "label": best["name"], "distracted": best["distracted"],
        "confidence": best["probability"], "low_confidence": best["probability"] < LOW_CONFIDENCE,
        "p_safe": p_safe, "p_distracted": 1 - p_safe,
        "probabilities": ranked,
    }


def classify_image(image):
    proba = get_model()(to_model_input(image)[None], training=False).numpy()[0]
    return {**_format(proba), "preview": preview_data_uri(image)}


def classify_bytes(data):
    return classify_image(open_image(data))


# --------------------------------------------------------------------------- #
# Sample frames (cropped from images/imageexample.png by scripts/build_metrics.py)
# --------------------------------------------------------------------------- #
def sample_names():
    return [s["file"] for s in get_metrics()["samples"]]


def load_sample(name):
    if name not in sample_names():
        raise KeyError(name)
    return Image.open(settings.SAMPLES_DIR / name).convert("RGB")
