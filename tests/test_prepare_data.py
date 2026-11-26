import pandas as pd
import pytest

from src.data_loader import METADATA_COLUMNS, load_driving_data
from src.prepare_data import prepare_dataset, split_records, validate_splits


def test_chronological_gaps_order_and_accounting(make_dataset):
    records = load_driving_data(make_dataset(count=100))
    splits, excluded, summary = split_records(records, gap=2)
    assert summary["strategy"] == "chronological"
    assert summary["counts"] == {"train": 68, "val": 11, "test": 13}
    assert len(excluded) == 8
    assert sum(len(part) for part in splits.values()) + len(excluded) == 100
    assert list(excluded["frame_index"]) == [68, 69, 70, 71, 83, 84, 85, 86]
    validate_splits(splits)
    all_rows = pd.concat([*splits.values(), excluded])
    assert all_rows["image_path"].nunique() == 100
    for part in splits.values():
        assert part["frame_index"].is_monotonic_increasing
        assert list(part.columns) == METADATA_COLUMNS


def test_sessions_are_disjoint_and_reproducible(make_dataset):
    records = load_driving_data(make_dataset(count=100))
    records["session_id"] = [f"drive_{index // 10}" for index in range(100)]
    splits, excluded, summary = split_records(records)
    repeated, _, _ = split_records(records)
    assert summary["strategy"] == "session"
    assert excluded.empty
    assert summary["counts"] == {"train": 70, "val": 10, "test": 20}
    validate_splits(splits, session_level=True)
    for name in splits:
        pd.testing.assert_frame_equal(splits[name], repeated[name])


def test_insufficient_sessions_are_not_claimed_independent(make_dataset):
    records = load_driving_data(make_dataset(count=100))
    records["session_id"] = "one_drive"
    _, _, summary = split_records(records)
    assert summary["strategy"] == "chronological"
    assert "NOT guaranteed" in summary["limitation"]
    with pytest.raises(ValueError, match="at least 3"):
        split_records(records, strategy="session")


def test_missing_session_ids_use_documented_fallback(make_dataset):
    records = load_driving_data(make_dataset(count=100))
    records["session_id"] = [f"drive_{index // 20}" for index in range(100)]
    records.loc[0, "session_id"] = None
    _, _, summary = split_records(records)
    assert summary["strategy"] == "chronological"


def test_small_dataset_and_duplicates_fail_clearly(make_dataset):
    records = load_driving_data(make_dataset(count=10))
    with pytest.raises(ValueError, match="Too few records"):
        split_records(records)
    with pytest.raises(ValueError, match="Duplicate"):
        split_records(pd.concat([records, records]))


def test_written_manifests_preserve_metadata(make_dataset, tmp_path):
    csv_path = make_dataset(count=100)
    output_dir = tmp_path / "processed"
    summary = prepare_dataset(csv_path, output_dir)
    for name, count in summary["counts"].items():
        manifest = pd.read_csv(output_dir / f"{name}.csv")
        assert len(manifest) == count
        assert list(manifest.columns) == METADATA_COLUMNS
    assert (output_dir / "split_summary.json").is_file()
    assert len(pd.read_csv(output_dir / "excluded_boundary.csv")) == 20


def test_existing_splits_reused_and_changes_refused(make_dataset, tmp_path):
    csv_path = make_dataset(count=100)
    output_dir = tmp_path / "processed"
    original = prepare_dataset(csv_path, output_dir)
    modified_time = (output_dir / "train.csv").stat().st_mtime_ns
    assert prepare_dataset(csv_path, output_dir) == original
    assert (output_dir / "train.csv").stat().st_mtime_ns == modified_time
    with pytest.raises(ValueError, match="Existing splits differ"):
        prepare_dataset(csv_path, output_dir, gap=1)
