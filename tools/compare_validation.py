"""Compare checkpoints on validation only, before opening the new test results."""

import argparse
from pathlib import Path

import numpy as np

from src.evaluate import regression_metrics
from src.pipeline import load_splits, make_dataset, write_json
from src.predict import load_trained_model


def compare_models(processed_dir, models, output):
    splits, summary, signatures = load_splits(processed_dir)
    actual = splits["val"]["steering"].to_numpy(dtype=float)
    results = []
    for path in models:
        model, metadata = load_trained_model(path)
        dataset = make_dataset(splits["val"], 32, metadata["preprocessing"])
        predicted = np.concatenate([np.asarray(model(images, training=False)).reshape(-1)
                                    for images, _ in dataset])
        metrics = regression_metrics(actual, predicted)
        results.append({"model_path": str(Path(path)), "model_sha256": metadata["model_sha256"],
                        **metrics, "mse": metrics["rmse"] ** 2,
                        "prediction_std": float(np.std(predicted)),
                        "original_training_images": metadata["counts"]["train"]})
    results.sort(key=lambda row: row["mse"])
    baseline = np.full_like(actual, splits["train"]["steering"].mean())
    comparison = {"selection_rule": "Lowest MSE on the same validation images; test predictions not computed.",
                  "validation_records": len(actual), "validation_signature": signatures["val"],
                  "baseline": regression_metrics(actual, baseline), "candidates": results,
                  "selected_model": results[0]["model_path"]}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write_json(output, comparison)
    for row in results:
        print(f"{row['model_path']}: validation MAE={row['mae']:.6f}, RMSE={row['rmse']:.6f}, "
              f"prediction std={row['prediction_std']:.6f}", flush=True)
    print(f"Validation-selected model: {comparison['selected_model']}", flush=True)
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    compare_models(**vars(parser.parse_args()))
