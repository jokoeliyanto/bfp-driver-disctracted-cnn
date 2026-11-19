"""Rebuild the published TensorFlow.js model as a Keras model.

The Bangkit team exported the final model to TensorFlow.js (models/tfjs/model.json + 13 weight shards). Rather than
ship a second 50 MB copy in another format, this module rebuilds the same architecture in Keras and copies the
weights straight from the shards, matching them by layer and variable name.

Architecture (from model.json): MobileNet v1 1.0/224 (ImageNet features, frozen during training)
-> Conv2D 1024 3x3 ReLU -> Dropout 0.2 -> GlobalAveragePooling -> Dense 10 softmax.

Preprocessing, verified against the weights (see scripts/build_metrics.py): RGB, 224 x 224, raw pixel values in
[0, 255]. Inputs rescaled to [0, 1] make the model predict the same class for every image.
"""
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

IMAGE_SIZE = 224


def read_weights(model_dir):
    """{"layer/variable": array} from the tfjs weight manifest."""
    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))["weightsManifest"]
    weights = {}
    for group in manifest:
        buf = b"".join((model_dir / p).read_bytes() for p in group["paths"])
        offset = 0
        for spec in group["weights"]:
            if spec["dtype"] != "float32":
                raise ValueError(f"Unsupported dtype {spec['dtype']} for {spec['name']}")
            count = int(np.prod(spec["shape"], dtype=np.int64))
            weights[spec["name"]] = np.frombuffer(buf, np.float32, count, offset).reshape(spec["shape"])
            offset += 4 * count
        if offset != len(buf):
            raise ValueError("Weight shards do not match the manifest.")
    return weights


def build_model(model_dir):
    """The Keras model with the published weights loaded."""
    import tensorflow as tf

    base = tf.keras.applications.MobileNet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights=None)
    model = tf.keras.Sequential([
        tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3)),
        base,
        tf.keras.layers.Conv2D(1024, 3, activation="relu", name="conv2d_8"),
        tf.keras.layers.Dropout(0.2, name="dropout_5"),
        tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d_8"),
        tf.keras.layers.Dense(10, activation="softmax", name="dense_8"),
    ], name="distracted_driver_mobilenet")

    weights = read_weights(model_dir)
    layers = [layer for layer in base.layers + model.layers
              if layer.weights and not isinstance(layer, tf.keras.Model)]
    used = 0
    for layer in layers:
        values = []
        for var in layer.weights:
            # Keras 3 calls a DepthwiseConv2D kernel "kernel"; Keras 2 (and the tfjs export) "depthwise_kernel"
            key = f"{layer.name}/{var.name}"
            if key not in weights:
                key = f"{layer.name}/depthwise_kernel"
            values.append(weights[key])
        layer.set_weights(values)
        used += len(values)
    if used != len(weights):
        raise ValueError(f"Loaded {used} of {len(weights)} weight tensors.")
    return model
