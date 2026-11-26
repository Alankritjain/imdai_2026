"""Analysis of synthetic fixtures stays exclusively inside pytest temporary folders."""

import json

import pytest

from src.explore_data import analyze_dataset


def test_analysis_artifacts_and_genuine_fixture_counts(make_dataset, tmp_path):
    csv_path = make_dataset(count=10)
    report_dir = tmp_path / "test_reports"
    summary = analyze_dataset(csv_path, report_dir, tmp_path / "processed",
                              dataset_source="SYNTHETIC PYTEST FIXTURE ONLY")
    assert summary["total_records"] == 10
    assert summary["valid_records"] == 10
    assert summary["steering"]["mean"] == pytest.approx(-0.05)
    assert summary["analysis_scope"] == "subset"
    assert summary["image_information"]["source_modes"] == {"RGB": 10}
    assert summary["image_information"]["dimensions_width_x_height"] == {"320x160": 10}
    assert json.loads((report_dir / "dataset_summary.json").read_text())["valid_records"] == 10
    for filename in ["steering_distribution.png", "sample_images.png", "preprocessing_comparison.png"]:
        assert (report_dir / "figures" / filename).stat().st_size > 0
