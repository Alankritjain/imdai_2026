import numpy as np
import pytest

from src.pipeline import load_splits, make_dataset as make_image_batches, preprocessing_settings
from src.prepare_data import prepare_dataset
from src.preprocess import load_image, preprocess_image
from src.pipeline import sampling_probabilities


def test_frozen_manifest_detects_copied_images(make_dataset, tmp_path):
    csv_path = make_dataset(count=50)
    processed = tmp_path / "processed"
    prepare_dataset(csv_path, processed, gap=1)
    splits, _, _ = load_splits(processed)
    from pathlib import Path
    Path(splits["test"].iloc[0].image_path).write_bytes(Path(splits["train"].iloc[0].image_path).read_bytes())
    with pytest.raises(ValueError, match="Duplicate image content"):
        load_splits(processed)


def test_validation_batches_are_ordered_unaugmented_and_repeatable(make_dataset, tmp_path):
    pytest.importorskip("tensorflow")
    processed = tmp_path / "processed"
    prepare_dataset(make_dataset(count=50), processed, gap=1)
    splits, _, _ = load_splits(processed)
    dataset = make_image_batches(splits["val"], 4, preprocessing_settings())
    first_images, first_labels = next(iter(dataset))
    repeated_images, repeated_labels = next(iter(dataset))
    row = splits["val"].iloc[0]
    assert np.array_equal(first_images[0], preprocess_image(load_image(row.image_path)))
    assert float(first_labels[0, 0]) == pytest.approx(row.steering)
    assert np.array_equal(first_images, repeated_images)
    assert np.array_equal(first_labels, repeated_labels)


def test_balancing_upsamples_turns_without_extreme_weights():
    labels = np.array([0.0] * 90 + [0.1] * 8 + [-0.8] * 2)
    probabilities = sampling_probabilities(labels)
    assert probabilities.sum() == pytest.approx(1)
    assert probabilities[-1] > probabilities[90] > probabilities[0]
    assert probabilities.max() / probabilities.min() <= 4.000001
    assert np.allclose(sampling_probabilities(np.zeros(5)), 0.2)


def test_balancing_forbidden_on_validation(make_dataset, tmp_path):
    pytest.importorskip("tensorflow")
    processed = tmp_path / "processed"
    prepare_dataset(make_dataset(count=40), processed, gap=1)
    splits, _, _ = load_splits(processed)
    with pytest.raises(ValueError, match="only for training"):
        make_image_batches(splits["val"], 4, preprocessing_settings(), balanced_sampling=True)
