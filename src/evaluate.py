"""Evaluate one fixed checkpoint against the training-mean baseline on held-out data."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import BATCH_SIZE, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from src.pipeline import file_sha256, load_splits, make_dataset, write_json
from src.predict import load_trained_model
from src.preprocess import load_image


def regression_metrics(actual, predicted):
    """Compute MAE/RMSE directly, independent of sklearn version changes."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("Metrics need matching nonempty one-dimensional arrays.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite targets and predictions.")
    error = predicted - actual
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2)))}


def save_evaluation_figures(predictions, figures_dir):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    actual, predicted = predictions["actual_steering"], predictions["predicted_steering"]
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(actual, predicted, alpha=0.65)
    lower, upper = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    if lower == upper:
        lower, upper = lower - 0.1, upper + 0.1
    axis.plot([lower, upper], [lower, upper], "k--", label="Perfect prediction")
    axis.set(xlabel="Actual steering (simulator units)", ylabel="CNN prediction (simulator units)",
             title="Held-out test: actual vs predicted")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures_dir / "actual_vs_predicted.png", dpi=150)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(predictions["prediction_error"], bins=20, edgecolor="white")
    axis.axvline(0, color="black", linestyle="--")
    axis.set(xlabel="Predicted minus actual (simulator units)", ylabel="Test records",
             title="Held-out prediction errors")
    figure.tight_layout()
    figure.savefig(figures_dir / "error_distribution.png", dpi=150)
    plt.close(figure)
    ordered = predictions.sort_values("absolute_error")
    examples = ordered.iloc[list(dict.fromkeys([0, min(1, len(ordered)-1), max(0, len(ordered)-2), len(ordered)-1]))]
    figure, axes = plt.subplots(2, 2, figsize=(10, 6))
    for axis in axes.flat:
        axis.axis("off")
    for axis, row in zip(axes.flat, examples.itertuples(index=False)):
        axis.imshow(load_image(row.image_path))
        axis.set_title(f"Actual {row.actual_steering:.4f} | CNN {row.predicted_steering:.4f}\n"
                       f"Absolute error {row.absolute_error:.4f}", fontsize=10)
    figure.suptitle("Held-out test examples: lower and higher errors")
    figure.tight_layout()
    figure.savefig(figures_dir / "sample_predictions.png", dpi=150)
    plt.close(figure)


def evaluate_model(processed_dir=PROCESSED_DIR, model_path=MODELS_DIR / "best_model.keras",
                   reports_dir=REPORTS_DIR, batch_size=BATCH_SIZE):
    """Use the same test rows for CNN and baseline; reject changed manifests."""
    model, metadata = load_trained_model(model_path)
    splits, summary, signatures = load_splits(processed_dir)
    if signatures != metadata["split_signatures"]:
        raise ValueError("Manifests/images changed since training; refusing a mismatched evaluation.")
    baseline_value = float(splits["train"]["steering"].mean())
    if not np.isclose(baseline_value, metadata["training_mean_steering"], rtol=0, atol=1e-12):
        raise ValueError("Training mean no longer matches the saved baseline.")
    dataset = make_dataset(splits["test"], batch_size, metadata["preprocessing"])
    outputs = []
    for images, _ in dataset:
        outputs.extend(np.asarray(model(images, training=False)).reshape(-1).tolist())
    actual = splits["test"]["steering"].to_numpy(dtype=float)
    predicted = np.asarray(outputs, dtype=float)
    metrics = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_sha256": metadata["model_sha256"], "split_signatures": signatures,
        "test_records": len(actual), "units": metadata["steering_units"],
        "cnn": regression_metrics(actual, predicted),
        "baseline": regression_metrics(actual, np.full_like(actual, baseline_value)),
        "baseline_training_mean": baseline_value,
        "split_limitation": summary["limitation"],
    }
    predictions = splits["test"][["image_path"]].copy()
    predictions["actual_steering"] = actual
    predictions["predicted_steering"] = predicted
    predictions["baseline_steering"] = baseline_value
    predictions["prediction_error"] = predicted - actual
    predictions["absolute_error"] = np.abs(predicted - actual)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    save_evaluation_figures(predictions, reports_dir / "figures")
    predictions.to_csv(reports_dir / "predictions.csv", index=False)
    metrics["predictions_sha256"] = file_sha256(reports_dir / "predictions.csv")
    write_json(reports_dir / "evaluation_metrics.json", metrics)
    for name in ("baseline", "cnn"):
        print(f"{name}: MAE={metrics[name]['mae']:.6f}, RMSE={metrics[name]['rmse']:.6f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--model-path", type=Path, default=MODELS_DIR / "best_model.keras")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    try:
        evaluate_model(**vars(parser.parse_args()))
    except (OSError, ValueError, ImportError, KeyError) as error:
        parser.exit(1, f"Evaluation: {error}\n")


if __name__ == "__main__":
    main()
