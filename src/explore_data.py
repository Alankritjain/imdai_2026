"""Inspect real driving records and save figures plus an auditable JSON summary."""

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures without needing a desktop window.
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import CROP_BOTTOM, CROP_TOP, DATA_DIR, DATASET_SOURCE, PROCESSED_DIR, REPORTS_DIR, STRAIGHT_THRESHOLD
from src.data_loader import dataset_fingerprint, find_driving_log, load_driving_data
from src.preprocess import crop_image, load_image, preprocess_image


def steering_statistics(records):
    """Population standard deviation; near-zero steering is not a universal category."""
    if records.empty:
        raise ValueError("No valid images with steering labels remain after validation.")
    steering = records["steering"]
    return {
        "minimum": float(steering.min()), "maximum": float(steering.max()),
        "mean": float(steering.mean()), "standard_deviation": float(steering.std(ddof=0)),
        "near_zero_threshold": STRAIGHT_THRESHOLD,
        "near_zero_percent": float((steering.abs() < STRAIGHT_THRESHOLD).mean() * 100),
    }


def steering_histogram(records):
    """Return a labelled figure; callers close it after displaying or saving."""
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.hist(records["steering"], bins=40, color="#137e88", edgecolor="white")
    axis.set(title="Center-camera steering distribution", xlabel="Steering value (simulator units)",
             ylabel="Number of valid records")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    return figure


def inspect_images(records, sample_size=100):
    """Inspect evenly spaced samples; do not claim these describe every image."""
    dimensions, formats, modes = Counter(), Counter(), Counter()
    indices = np.linspace(0, len(records) - 1, min(sample_size, len(records)), dtype=int)
    for index in indices:
        with Image.open(records.iloc[index]["image_path"]) as image:
            dimensions[f"{image.width}x{image.height}"] += 1
            formats[str(image.format)] += 1
            modes[image.mode] += 1
    return {"inspected_images": len(indices), "sampling": "evenly spaced valid rows",
            "dimensions_width_x_height": dict(dimensions), "formats": dict(formats),
            "source_modes": dict(modes), "consistent_dimensions_in_sample": len(dimensions) == 1,
            "processed_channels": "RGB (3 channels)"}


def save_sample_images(records, output_path):
    # Include steering extrema for illustration; the histogram describes frequency.
    steering = records["steering"].to_numpy()
    candidates = [int(steering.argmin()), int(steering.argmax())]
    candidates += np.linspace(0, len(records) - 1, min(6, len(records)), dtype=int).tolist()
    indices = list(dict.fromkeys(candidates))[:6]
    figure, axes = plt.subplots(2, 3, figsize=(12, 5))
    for axis in axes.flat:
        axis.axis("off")
    for axis, index in zip(axes.flat, indices):
        record = records.iloc[index]
        axis.imshow(load_image(record["image_path"]))
        axis.set_title(f"Ground truth: {record['steering']:.4f}")
    figure.suptitle("Actual center-camera examples (including steering extrema)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_preprocessing_comparison(image_path, output_path):
    original = load_image(image_path)
    stages = [original, crop_image(original), preprocess_image(original)]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = ["Original (RGB)", f"Cropped ({CROP_TOP:.0%} top, {CROP_BOTTOM:.0%} bottom)",
              f"Processed {stages[2].shape}, [0, 1]"]
    for axis, stage, label in zip(axes, stages, labels):
        axis.imshow(stage)  # Matplotlib displays float RGB arrays in [0, 1] directly.
        axis.set_title(label, fontsize=10)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def analyze_dataset(csv_path, reports_dir=REPORTS_DIR, processed_dir=PROCESSED_DIR,
                    scope="subset", dataset_source=DATASET_SOURCE):
    """Analyze one log. Defaults to subset: a selected log is not the entire download."""
    records = load_driving_data(csv_path)
    statistics = steering_statistics(records)
    reports_dir, processed_dir = Path(reports_dir), Path(processed_dir)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        **records.attrs["validation"], "dataset_source": dataset_source,
        "dataset_path": str(Path(csv_path).resolve().parent), "analysis_scope": scope,
        "scope_note": "Statistics cover valid records from the selected CSV only.",
        "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": dataset_fingerprint(records),
        "steering": statistics, "image_information": inspect_images(records),
        "preprocessing": {"crop_top_fraction": CROP_TOP, "crop_bottom_fraction": CROP_BOTTOM,
                          "normalization": "RGB float32 / 255.0"},
    }
    provenance_path = Path(csv_path).parent / "provenance.json"
    if provenance_path.exists():
        summary["download_provenance"] = json.loads(provenance_path.read_text(encoding="utf-8"))
        if scope == "complete" and summary["download_provenance"].get("analysis_scope") == "subset":
            raise ValueError("The download provenance identifies a subset; do not label it complete.")
    figure = steering_histogram(records)
    figure.savefig(figures_dir / "steering_distribution.png", dpi=150)
    plt.close(figure)
    save_sample_images(records, figures_dir / "sample_images.png")
    save_preprocessing_comparison(records.iloc[0]["image_path"], figures_dir / "preprocessing_comparison.png")
    records.to_csv(processed_dir / "cleaned_metadata.csv", index=False)
    (reports_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--csv", type=Path, help="Explicit log path, relative to the current directory")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--scope", choices=["subset", "complete"], default="subset",
                        help="Use complete only after verifying full dataset coverage")
    arguments = parser.parse_args()
    try:
        selected_csv = find_driving_log(arguments.data_dir, arguments.csv)
        summary = analyze_dataset(selected_csv, arguments.reports_dir, arguments.processed_dir, arguments.scope)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Dataset analysis: {error}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
