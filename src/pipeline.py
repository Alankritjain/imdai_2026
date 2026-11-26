"""Small shared helpers for frozen manifests and memory-efficient image batches."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from config import CROP_BOTTOM, CROP_TOP, IMAGE_HEIGHT, IMAGE_WIDTH, RANDOM_SEED
from src.prepare_data import validate_splits
from src.preprocess import adjust_brightness, flip_image, load_image, preprocess_image


def preprocessing_settings():
    return {"crop_top": CROP_TOP, "crop_bottom": CROP_BOTTOM,
            "width": IMAGE_WIDTH, "height": IMAGE_HEIGHT, "channels": "RGB",
            "normalization": "float32 / 255.0"}


def file_sha256(path):
    """Hash a file without loading the entire model into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_splits(processed_dir):
    """Check frozen splits, including copied images with different filenames.

    Test metadata is checked for separation only; training never uses its targets.
    Signatures use image bytes and labels, so moving the project does not change them.
    """
    processed_dir = Path(processed_dir)
    summary = json.loads((processed_dir / "split_summary.json").read_text(encoding="utf-8"))
    splits, signatures, seen_content = {}, {}, set()
    for name in ("train", "val", "test"):
        records = pd.read_csv(processed_dir / f"{name}.csv", keep_default_na=False)
        required = {"image_path", "steering", "frame_index", "source_csv", "session_id"}
        if records.empty or not required.issubset(records.columns):
            raise ValueError(f"Invalid {name} manifest. Run src.prepare_data first.")
        records["steering"] = pd.to_numeric(records["steering"], errors="raise")
        if not np.isfinite(records["steering"]).all():
            raise ValueError(f"Non-finite steering in {name} manifest.")
        if len(records) != summary["counts"][name]:
            raise ValueError(f"{name} manifest differs from split_summary.json.")
        digest = hashlib.sha256()
        for row in records.itertuples(index=False):
            path = Path(row.image_path)
            if not path.is_file():
                raise FileNotFoundError(f"Image missing: {path}. On a new machine, prepare manifests locally.")
            content_hash = file_sha256(path)
            if content_hash in seen_content:
                raise ValueError("Duplicate image content within/across splits. Audit data before training.")
            seen_content.add(content_hash)
            payload = [content_hash, float(row.steering), int(row.frame_index), str(row.session_id)]
            digest.update(json.dumps(payload).encode("utf-8"))
        splits[name], signatures[name] = records, digest.hexdigest()
    validate_splits(splits, session_level=summary["strategy"] == "session")
    if summary["strategy"] == "chronological":
        combined = pd.concat(splits.values())
        if combined["source_csv"].nunique() != 1:
            raise ValueError("Chronological splits require a single recording log.")
        for left, right in (("train", "val"), ("val", "test")):
            if not splits[left]["frame_index"].is_monotonic_increasing:
                raise ValueError("Chronological manifests must preserve frame order.")
            separation = splits[right]["frame_index"].min() - splits[left]["frame_index"].max()
            if separation <= 2 * summary["gap_each_side"]:
                raise ValueError("Chronological split boundaries overlap or lack the recorded gap.")
    return splits, summary, signatures


def sampling_probabilities(steering):
    """Moderately upsample turns using training labels only, without extreme weights."""
    magnitudes = np.abs(np.asarray(steering, dtype=float))
    groups = np.digitize(magnitudes, [0.05, 0.25])
    counts = np.bincount(groups, minlength=3)
    weights = 1.0 / np.sqrt(counts[groups])
    weights = np.minimum(weights, weights.min() * 4)
    return weights / weights.sum()


def make_dataset(records, batch_size, settings, training=False, seed=RANDOM_SEED,
                 balanced_sampling=False):
    """Load one image at a time; buffer at most a few batches, never all images."""
    import tensorflow as tf

    if batch_size < 1 or records.empty:
        raise ValueError("Batch size and dataset size must be positive.")
    random = np.random.default_rng(seed)
    if balanced_sampling and not training:
        raise ValueError("Balanced sampling is allowed only for training data.")
    probabilities = sampling_probabilities(records["steering"]) if balanced_sampling else None

    def examples():
        if balanced_sampling:
            indices = random.choice(len(records), size=len(records), replace=True, p=probabilities)
        else:
            indices = random.permutation(len(records)) if training else np.arange(len(records))
        for index in indices:
            row = records.iloc[index]
            image, steering = load_image(row["image_path"]), float(row["steering"])
            if training:
                if random.random() < 0.5:
                    image, steering = flip_image(image, steering)
                image = adjust_brightness(image, float(random.uniform(0.8, 1.2)))
            pixels = preprocess_image(image, settings["crop_top"], settings["crop_bottom"])
            yield pixels, np.array([steering], dtype=np.float32)

    dataset = tf.data.Dataset.from_generator(examples, output_signature=(
        tf.TensorSpec((IMAGE_HEIGHT, IMAGE_WIDTH, 3), tf.float32),
        tf.TensorSpec((1,), tf.float32)))
    dataset = dataset.batch(batch_size).apply(
        tf.data.experimental.assert_cardinality(math.ceil(len(records) / batch_size)))
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    return dataset.with_options(options).prefetch(1)
