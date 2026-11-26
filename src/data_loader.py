"""Read one explicitly selected simulator log without loading images into RAM."""

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
METADATA_COLUMNS = ["image_path", "steering", "source_csv", "frame_index", "session_id"]
CENTER_COLUMNS = {"center", "centre", "center_image", "center_camera", "image_path"}
STEERING_COLUMNS = {"steering", "steering_angle", "steering_value"}
SESSION_COLUMNS = {"session_id", "drive_id", "recording_id"}


def find_driving_logs(data_dir):
    """Discover logs; never silently merge tracks and their combined copies."""
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    return sorted(path for path in data_dir.rglob("*")
                  if path.is_file() and path.name.lower() == "driving_log.csv")


def find_driving_log(data_dir, csv_path=None):
    """Use --csv when multiple logs exist, or when a log has another filename."""
    if csv_path is not None:
        selected = Path(csv_path).expanduser().resolve()
        if not selected.is_file():
            raise FileNotFoundError(f"CSV not found: {selected}")
        return selected
    logs = find_driving_logs(data_dir)
    if not logs:
        raise FileNotFoundError(
            f"No driving_log.csv in {Path(data_dir).resolve()}. Download and extract "
            "zaynena/selfdriving-car-simulator, or pass --csv with your log path."
        )
    if len(logs) > 1:
        choices = "\n".join(str(path) for path in logs)
        raise ValueError(f"Multiple logs found. Select exactly one with --csv:\n{choices}")
    return logs[0]


def identify_columns(first_row):
    """Recognize common headers, otherwise use Udacity's center/left/right/steering order."""
    names = [value.strip().lower().replace(" ", "_") for value in first_row]
    has_header = bool(set(names) & (CENTER_COLUMNS | STEERING_COLUMNS | SESSION_COLUMNS))
    if has_header:
        centers = [index for index, name in enumerate(names) if name in CENTER_COLUMNS]
        steering = [index for index, name in enumerate(names) if name in STEERING_COLUMNS]
        sessions = [index for index, name in enumerate(names) if name in SESSION_COLUMNS]
        if len(centers) != 1 or len(steering) != 1 or len(sessions) > 1:
            raise ValueError("CSV needs one center image column and one steering column; "
                             "optional session_id/drive_id/recording_id must be unique.")
        return True, centers[0], steering[0], sessions[0] if sessions else None
    # An all-word first row is probably an unsupported header, not a driving sample.
    if names and all(name.replace("_", "").isalpha() for name in names):
        raise ValueError("Unrecognized CSV header. Expected center and steering columns.")
    if len(first_row) < 4:
        raise ValueError("Headerless CSV must use center,left,right,steering[,throttle,brake,speed].")
    return False, 0, 3, None


def build_image_index(csv_directory):
    """Index filenames only within the selected log's directory tree."""
    image_index = {}
    for path in csv_directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            image_index.setdefault(path.name.casefold(), []).append(path.resolve())
    return image_index


def resolve_image_path(raw_path, csv_directory, image_index):
    """Recover foreign absolute paths only when the local filename is unambiguous."""
    normalized = raw_path.strip().replace("\\", "/")
    if not normalized:
        return None, "missing_image_path"
    image_path = Path(normalized)
    if image_path.suffix.lower() not in IMAGE_SUFFIXES:
        return None, "unsupported_image"
    direct_path = image_path if image_path.is_absolute() else csv_directory / image_path
    if direct_path.is_file():
        return direct_path.resolve(), None
    candidates = image_index.get(image_path.name.casefold(), [])
    if len(candidates) == 1:
        return candidates[0], None
    if len(candidates) > 1:
        return None, "ambiguous_image_path"
    return None, "missing_image"


def load_driving_data(csv_path):
    """Return valid metadata, with validation counts in DataFrame.attrs['validation'].

    First valid occurrence of an image wins. Invalid rows are counted once by their
    first failure. frame_index preserves original CSV order (zero-based, no header).
    No session is inferred merely from a directory or CSV filename.
    """
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    image_index = build_image_index(csv_path.parent)
    valid_records = []
    seen_images = set()
    seen_content = set()
    invalid_reasons = Counter()
    total_records = 0
    columns = None
    try:
        with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                if not row or not any(value.strip() for value in row):
                    continue
                if columns is None:
                    has_header, center_column, steering_column, session_column = identify_columns(row)
                    columns = (center_column, steering_column, session_column)
                    if has_header:
                        continue
                frame_index = total_records
                total_records += 1
                required_columns = [column for column in columns if column is not None]
                if len(row) <= max(required_columns):
                    invalid_reasons["malformed_row"] += 1
                    continue
                raw_steering = row[steering_column].strip()
                if not raw_steering:
                    invalid_reasons["missing_steering"] += 1
                    continue
                try:
                    steering_value = float(raw_steering)
                except ValueError:
                    invalid_reasons["non_numeric_steering"] += 1
                    continue
                if not math.isfinite(steering_value):
                    invalid_reasons["non_finite_steering"] += 1
                    continue
                image_path, problem = resolve_image_path(row[center_column], csv_path.parent, image_index)
                if problem:
                    invalid_reasons[problem] += 1
                    continue
                image_key = str(image_path).casefold()
                if image_key in seen_images:
                    invalid_reasons["duplicate_image"] += 1
                    continue
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                    # Decode pixels too: verify() alone may miss truncated image data.
                    with Image.open(image_path) as image:
                        image.convert("RGB").load()
                except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
                    invalid_reasons["unreadable_image"] += 1
                    continue
                content_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
                if content_hash in seen_content:
                    invalid_reasons["duplicate_content"] += 1
                    continue
                seen_images.add(image_key)
                seen_content.add(content_hash)
                valid_records.append({
                    "image_path": str(image_path), "steering": steering_value,
                    "source_csv": str(csv_path), "frame_index": frame_index,
                    "session_id": row[session_column].strip() if session_column is not None else "",
                })
    except (csv.Error, UnicodeError) as error:
        raise ValueError(f"Cannot read CSV {csv_path}: {error}. Use a UTF-8 comma-separated file.") from error
    if columns is None:
        raise ValueError(f"CSV is empty: {csv_path}")
    records = pd.DataFrame(valid_records, columns=METADATA_COLUMNS)
    records.attrs["validation"] = {
        "total_records": total_records, "valid_records": len(records),
        "invalid_records": sum(invalid_reasons.values()),
        "missing_images": invalid_reasons["missing_image"],
        "duplicate_records": invalid_reasons["duplicate_image"] + invalid_reasons["duplicate_content"],
        "invalid_reasons": dict(invalid_reasons), "selected_csv": str(csv_path),
    }
    return records


def dataset_fingerprint(records):
    """Identify the selected metadata so the dashboard does not show stale progress."""
    payload = records.to_csv(index=False) + json.dumps(records.attrs.get("validation", {}), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
