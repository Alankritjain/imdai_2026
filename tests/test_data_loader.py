import pytest
from PIL import Image

from src.data_loader import find_driving_log, load_driving_data


def test_header_whitespace_and_windows_path(make_dataset):
    csv_path = make_dataset(rows=[[r" C:\old\drive\IMG\center_0.jpg ", " 0.25 "]],
                            header=[" center ", " steering "], count=1)
    records = load_driving_data(csv_path)
    assert len(records) == 1
    assert records.iloc[0]["steering"] == 0.25
    assert records.iloc[0]["image_path"] == str((csv_path.parent / "IMG/center_0.jpg").resolve())


def test_headerless_first_record_is_not_lost(make_dataset):
    csv_path = make_dataset(rows=[["IMG/center_0.jpg", "left.jpg", "right.jpg", "-0.3", 0, 0, 4]],
                            header=None, count=1)
    records = load_driving_data(csv_path)
    assert len(records) == 1
    assert records.iloc[0]["steering"] == -0.3
    assert records.iloc[0]["frame_index"] == 0


def test_invalid_rows_and_duplicates_are_counted_once(make_dataset):
    rows = [["IMG/center_0.jpg", 0], ["IMG/center_0.jpg", 0.1], ["", 0],
            ["IMG/center_1.jpg", ""], ["IMG/center_1.jpg", "bad"],
            ["IMG/center_1.jpg", "nan"], ["IMG/center_1.jpg", "inf"],
            ["missing.jpg", 0], ["file.txt", 0], ["too_short"], ["IMG/center_1.jpg", -0.2]]
    records = load_driving_data(make_dataset(rows=rows, count=2))
    report = records.attrs["validation"]
    assert len(records) == 2
    assert report["total_records"] == 11
    assert report["invalid_records"] == 9
    assert report["missing_images"] == 1
    assert report["duplicate_records"] == 1
    assert report["invalid_reasons"]["non_finite_steering"] == 2
    assert sum(report["invalid_reasons"].values()) + len(records) == 11


def test_ambiguous_foreign_filename_is_rejected(make_dataset):
    csv_path = make_dataset(rows=[[r"Z:\other\center_0.jpg", 0]], count=1)
    another_dir = csv_path.parent / "other/IMG"
    another_dir.mkdir(parents=True)
    Image.new("RGB", (20, 20)).save(another_dir / "center_0.jpg")
    records = load_driving_data(csv_path)
    assert records.empty
    assert records.attrs["validation"]["invalid_reasons"]["ambiguous_image_path"] == 1


def test_corrupt_image_is_rejected(make_dataset):
    csv_path = make_dataset(count=1)
    (csv_path.parent / "IMG/center_0.jpg").write_text("not a JPEG")
    records = load_driving_data(csv_path)
    assert records.empty
    assert records.attrs["validation"]["invalid_reasons"]["unreadable_image"] == 1


def test_multiple_logs_require_explicit_choice(make_dataset):
    csv_path = make_dataset(count=1)
    second = csv_path.parent / "combined/driving_log.csv"
    second.parent.mkdir()
    second.write_text(csv_path.read_text())
    with pytest.raises(ValueError, match="Multiple logs"):
        find_driving_log(csv_path.parent)
    assert find_driving_log(csv_path.parent, csv_path) == csv_path


def test_missing_steering_column_has_meaningful_error(make_dataset):
    csv_path = make_dataset(header=["center", "speed"], count=1)
    with pytest.raises(ValueError, match="steering column"):
        load_driving_data(csv_path)


def test_missing_log_and_empty_csv(tmp_path):
    with pytest.raises(FileNotFoundError, match="No driving_log"):
        find_driving_log(tmp_path)
    empty_csv = tmp_path / "driving_log.csv"
    empty_csv.touch()
    with pytest.raises(ValueError, match="empty"):
        load_driving_data(empty_csv)


def test_session_ids_are_preserved(make_dataset):
    csv_path = make_dataset(rows=[["IMG/center_0.jpg", 0.1, " drive_A "]],
                            header=["center", "steering", "session_id"], count=1)
    records = load_driving_data(csv_path)
    assert records.iloc[0]["session_id"] == "drive_A"


def test_copied_image_is_rejected_even_with_different_name(make_dataset):
    csv_path = make_dataset(count=2)
    (csv_path.parent / "IMG/center_1.jpg").write_bytes((csv_path.parent / "IMG/center_0.jpg").read_bytes())
    records = load_driving_data(csv_path)
    assert len(records) == 1
    assert records.attrs["validation"]["invalid_reasons"]["duplicate_content"] == 1
