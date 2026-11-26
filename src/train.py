"""Train the milestone CNN on frozen manifests and save verified real artifacts."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (BATCH_SIZE, EARLY_STOPPING_PATIENCE, LEARNING_RATE, MAX_EPOCHS,
                    MODELS_DIR, PROCESSED_DIR, RANDOM_SEED, REPORTS_DIR)
from src.model import build_model
from src.pipeline import (file_sha256, load_splits, make_dataset, preprocessing_settings, write_json)
from src.predict import predict_steering


def train_model(processed_dir=PROCESSED_DIR, models_dir=MODELS_DIR, reports_dir=REPORTS_DIR,
                epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                patience=EARLY_STOPPING_PATIENCE, seed=RANDOM_SEED, balanced_sampling=False,
                reduce_lr=False, warmup_epochs=0):
    """Fit train data, select with validation only, and leave test inference untouched."""
    import tensorflow as tf

    if epochs < 1 or batch_size < 1 or patience < 0 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Epochs, batch size and learning rate must be positive; patience nonnegative.")
    if warmup_epochs < 0 or warmup_epochs >= epochs:
        raise ValueError("Warmup epochs must be between zero and maximum epochs minus one.")
    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    if (models_dir / "best_model.keras").exists():
        raise FileExistsError("A model already exists. Use a new --models-dir and --reports-dir for another experiment.")
    splits, split_summary, signatures = load_splits(processed_dir)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    settings = preprocessing_settings()
    training = make_dataset(splits["train"], batch_size, settings, training=True, seed=seed,
                            balanced_sampling=balanced_sampling)
    validation = make_dataset(splits["val"], batch_size, settings)
    model = build_model(learning_rate)
    model.summary()
    print(f"Parameters: {model.count_params():,}; train={len(splits['train'])}, val={len(splits['val'])}")
    models_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)
    checkpoint = models_dir / "best_model.keras"

    class FiniteLoss(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if not all(np.isfinite((logs or {}).get(key, np.nan)) for key in ("loss", "val_loss")):
                raise ValueError("Non-finite training/validation loss; run is not complete.")

    callbacks = [FiniteLoss(),
                 tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", save_best_only=True),
                 tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                                  start_from_epoch=warmup_epochs, restore_best_weights=True)]
    if reduce_lr:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1))
    history = model.fit(training, validation_data=validation, epochs=epochs, callbacks=callbacks,
                        shuffle=False, verbose=2)  # The generator already shuffles training rows.
    history_frame = pd.DataFrame(history.history)
    history_frame.insert(0, "epoch", np.arange(1, len(history_frame) + 1))
    best_model = tf.keras.models.load_model(checkpoint, compile=False)
    verification = predict_steering(splits["val"].iloc[0]["image_path"], best_model, settings)
    validation_predictions = np.concatenate([
        np.asarray(best_model(images, training=False)).reshape(-1) for images, _ in validation])
    validation_frame = splits["val"][["image_path", "steering"]].copy()
    validation_frame["predicted_steering"] = validation_predictions
    validation_frame.to_csv(reports_dir / "validation_predictions.csv", index=False)
    history_frame.to_csv(reports_dir / "training_history.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(history_frame["epoch"], history_frame["loss"], label="Training (augmented)")
    axis.plot(history_frame["epoch"], history_frame["val_loss"], label="Validation")
    axis.set(xlabel="Epoch", ylabel="MSE (simulator units squared)", title="Training and validation loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(reports_dir / "figures" / "training_loss.png", dpi=150)
    plt.close(figure)
    metadata = {
        "training_status": "complete", "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_sha256": file_sha256(checkpoint), "parameter_count": model.count_params(),
        "preprocessing": settings,
        "configuration": {"optimizer": "Adam", "loss": "MSE", "learning_rate": learning_rate,
                          "batch_size": batch_size, "maximum_epochs": epochs, "patience": patience, "seed": seed,
                          "balanced_sampling": balanced_sampling, "reduce_lr": reduce_lr,
                          "warmup_epochs": warmup_epochs},
        "augmentation": {"training_only": True, "flip_probability": 0.5, "brightness_factor": [0.8, 1.2]},
        "epochs_completed": len(history_frame), "best_epoch": int(history_frame["val_loss"].idxmin() + 1),
        "best_validation_loss": float(history_frame["val_loss"].min()),
        "validation_mae": float(np.mean(np.abs(validation_predictions - splits["val"]["steering"].to_numpy()))),
        "validation_prediction_std": float(np.std(validation_predictions)),
        "counts": {name: len(records) for name, records in splits.items()},
        "training_mean_steering": float(splits["train"]["steering"].mean()),
        "validation_baseline_mse": float(np.mean((splits["val"]["steering"] - splits["train"]["steering"].mean()) ** 2)),
        "split_signatures": signatures, "split_summary": split_summary,
        "steering_units": "native simulator units (not degrees)", "steering_sign_convention": "unverified",
        "direction_note": "Direction labels disabled: this dataset's sign convention has not been independently verified.",
        "reload_verification_prediction": verification,
        "runtime": {"python": platform.python_version(), "tensorflow": tf.__version__,
                    "keras": tf.keras.__version__, "devices": [device.name for device in tf.config.list_physical_devices()]},
        "history_sha256": file_sha256(reports_dir / "training_history.csv"),
    }
    write_json(models_dir / "model_metadata.json", metadata)
    print(f"Verified checkpoint: {checkpoint}; best validation MSE={metadata['best_validation_loss']:.6f}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--balanced-sampling", action="store_true", help="Moderately upsample turns in training only")
    parser.add_argument("--reduce-lr", action="store_true", help="Reduce learning rate when validation loss plateaus")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Delay early stopping, but still checkpoint every epoch")
    arguments = parser.parse_args()
    try:
        train_model(**vars(arguments))
    except (OSError, ValueError, ImportError, KeyError) as error:
        parser.exit(1, f"Training: {error}\n")


if __name__ == "__main__":
    main()
