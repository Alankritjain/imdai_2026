"""Reload a trained model and its exact preprocessing contract for inference."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from config import MODELS_DIR, PROCESSED_DIR
from src.pipeline import file_sha256
from src.preprocess import preprocess_image


def load_trained_model(model_path=MODELS_DIR / "best_model.keras"):
    """Require matching metadata; never silently use unknown preprocessing."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError("Trained model not found. Please complete model training before using steering prediction.")
    metadata = json.loads((model_path.parent / "model_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("training_status") != "complete" or file_sha256(model_path) != metadata.get("model_sha256"):
        raise ValueError("Model and metadata do not match a completed training run.")
    settings = metadata["preprocessing"]
    if (settings["width"], settings["height"], settings["channels"], settings["normalization"]) != (
            160, 80, "RGB", "float32 / 255.0"):
        raise ValueError("Unsupported saved preprocessing settings.")
    from tensorflow import keras
    model = keras.models.load_model(model_path, compile=False)
    if model.input_shape != (None, 80, 160, 3) or model.output_shape != (None, 1):
        raise ValueError("Saved model has incompatible input/output shapes.")
    # Store the saved settings on the loaded model for the simple two-argument API.
    model.steering_preprocessing = settings
    return model, metadata


def predict_steering(image, model, settings=None):
    """Accept a PIL image, file path or uploaded file and return one finite float."""
    settings = settings or getattr(model, "steering_preprocessing", None)
    if settings is None:
        raise ValueError("Saved preprocessing settings are required. Use load_trained_model.")
    if isinstance(image, Image.Image):
        pixels = preprocess_image(image, settings["crop_top"], settings["crop_bottom"])
    else:
        with Image.open(image) as opened:
            pixels = preprocess_image(opened, settings["crop_top"], settings["crop_bottom"])
    prediction = np.asarray(model(pixels[np.newaxis, ...], training=False))
    if prediction.shape != (1, 1) or not np.isfinite(prediction).all():
        raise ValueError("Model must return exactly one finite steering value.")
    return float(prediction[0, 0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, nargs="?", help="Optional image; otherwise use a built-in held-out frame")
    parser.add_argument("--sample-index", type=int, default=0, help="Zero-based held-out frame when no image is supplied")
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "best_model.keras")
    arguments = parser.parse_args()
    try:
        image_path = arguments.image
        if image_path is None:
            import pandas as pd
            samples = pd.read_csv(PROCESSED_DIR / "test.csv")
            if not 0 <= arguments.sample_index < len(samples):
                raise ValueError(f"Sample index must be between 0 and {len(samples) - 1}.")
            image_path = samples.iloc[arguments.sample_index]["image_path"]
        model, _ = load_trained_model(arguments.model)
        print(f"Image: {image_path}")
        print(f"Predicted Steering: {predict_steering(image_path, model):.6f} (simulator units)")
    except (OSError, ValueError, ImportError, KeyError) as error:
        parser.exit(1, f"Prediction: {error}\n")


if __name__ == "__main__":
    main()
