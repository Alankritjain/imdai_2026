"""Create metadata manifests using complete sessions or chronological blocks."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import BOUNDARY_GAP, DATA_DIR, PROCESSED_DIR, RANDOM_SEED, SPLIT_RATIOS
from src.data_loader import METADATA_COLUMNS, dataset_fingerprint, find_driving_log, load_driving_data


def validate_splits(splits, session_level=False):
    """Fail before saving if an image (or an identified session) crosses a boundary."""
    seen_images, seen_sessions = set(), set()
    for name, records in splits.items():
        if records.empty:
            raise ValueError(f"The {name} split is empty. Supply more records/sessions or reduce --gap.")
        images = set(records["image_path"].map(lambda path: str(Path(path).resolve()).casefold()))
        if len(images) != len(records) or seen_images & images:
            raise ValueError("Duplicate or overlapping images found in dataset splits.")
        seen_images.update(images)
        if session_level:
            sessions = set(records["session_id"])
            if seen_sessions & sessions:
                raise ValueError("A recording session appears in more than one split.")
            seen_sessions.update(sessions)


def split_records(records, strategy="auto", gap=BOUNDARY_GAP, seed=RANDOM_SEED):
    """Return train/val/test frames, excluded boundary rows, and a strategy summary.

    Session ratios apply to session counts; unequal session lengths change row ratios.
    Chronological fallback assumes original log order is recording order. It reduces
    immediate neighbor leakage but cannot establish independent drives or routes.
    """
    if strategy not in {"auto", "session", "chronological"}:
        raise ValueError("Strategy must be auto, session or chronological.")
    if not isinstance(gap, int) or gap < 0:
        raise ValueError("Boundary gap must be a nonnegative integer.")
    if records.empty or not set(METADATA_COLUMNS).issubset(records.columns):
        raise ValueError("Supply valid loader metadata with image, steering, source and session fields.")
    if records["image_path"].map(lambda path: str(Path(path).resolve()).casefold()).duplicated().any():
        raise ValueError("Duplicate images must be removed before splitting.")
    session_ids = records["session_id"].fillna("").astype(str).str.strip()
    sessions_complete = bool(session_ids.ne("").all())
    session_count = session_ids[session_ids.ne("")].nunique()
    can_split_sessions = sessions_complete and session_count >= 3
    if strategy == "session" and not can_split_sessions:
        raise ValueError("Session splitting needs at least 3 identified sessions and no missing session IDs.")
    use_sessions = can_split_sessions and strategy != "chronological"
    records = records.copy()
    records["session_id"] = session_ids
    excluded = records.iloc[:0].copy()
    if use_sessions:
        sessions = np.random.default_rng(seed).permutation(sorted(session_ids.unique()))
        train_count = max(1, min(len(sessions) - 2, int(len(sessions) * SPLIT_RATIOS[0])))
        val_count = max(1, min(len(sessions) - train_count - 1, int(len(sessions) * SPLIT_RATIOS[1])))
        groups = [sessions[:train_count], sessions[train_count:train_count + val_count],
                  sessions[train_count + val_count:]]
        splits = {name: records[records["session_id"].isin(group)].copy()
                  for name, group in zip(("train", "val", "test"), groups)}
        limitation = ("Whole sessions are disjoint. Ratios target session counts, not row counts; "
                      "few or unequal sessions can differ greatly from 70/15/15. "
                      "Repeated routes and environments can still be correlated.")
    else:
        if records["source_csv"].nunique() != 1:
            raise ValueError("Chronological fallback requires one log; select a log or supply session IDs.")
        # Sort by original row index, never shuffle individual frames.
        records = records.sort_values("frame_index", kind="stable")
        first_boundary = int(len(records) * SPLIT_RATIOS[0])
        second_boundary = int(len(records) * (SPLIT_RATIOS[0] + SPLIT_RATIOS[1]))
        if (first_boundary <= gap or second_boundary - first_boundary <= 2 * gap
                or len(records) - second_boundary <= gap):
            raise ValueError("Too few records for three nonempty blocks with this gap. "
                             "Use more data or explicitly choose a smaller --gap.")
        splits = {
            "train": records.iloc[:first_boundary - gap].copy(),
            "val": records.iloc[first_boundary + gap:second_boundary - gap].copy(),
            "test": records.iloc[second_boundary + gap:].copy(),
        }
        excluded = pd.concat([records.iloc[first_boundary - gap:first_boundary + gap],
                              records.iloc[second_boundary - gap:second_boundary + gap]]).copy()
        limitation = ("Fewer than three fully identified sessions, or chronological strategy requested. "
                      "CSV row order is assumed chronological and must be checked against timestamps. "
                      "Boundary gaps reduce neighboring-frame leakage, but session independence is NOT "
                      "guaranteed. Repeated scenery and longer temporal correlations may cross splits.")
    validate_splits(splits, session_level=use_sessions)
    counts = {name: len(part) for name, part in splits.items()}
    if sum(counts.values()) + len(excluded) != len(records):
        raise ValueError("Split accounting does not match the input record count.")
    summary = {
        "strategy": "session" if use_sessions else "chronological",
        "seed": seed, "requested_ratios": dict(zip(("train", "val", "test"), SPLIT_RATIOS)),
        "input_records": len(records), "counts": counts, "excluded_boundary_records": len(excluded),
        "gap_each_side": 0 if use_sessions else gap, "identified_sessions": int(session_count),
        "retained_percentages": {name: count / sum(counts.values()) * 100 for name, count in counts.items()},
        "sessions_per_split": {name: sorted(set(part["session_id"]) - {""}) for name, part in splits.items()},
        "limitation": limitation, "dataset_fingerprint": dataset_fingerprint(records),
        "selected_csvs": sorted(records["source_csv"].unique().tolist()),
    }
    return splits, excluded, summary


def prepare_dataset(csv_path, output_dir=PROCESSED_DIR, strategy="auto", gap=BOUNDARY_GAP):
    records = load_driving_data(csv_path)
    splits, excluded, summary = split_records(records, strategy, gap)
    # Fingerprint the loader's original order and validation report.
    summary["dataset_fingerprint"] = dataset_fingerprint(records)
    summary["prepared_at_utc"] = datetime.now(timezone.utc).isoformat()
    output_dir = Path(output_dir)
    summary_path = output_dir / "split_summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        same_settings = all(previous.get(key) == summary.get(key)
                            for key in ("dataset_fingerprint", "strategy", "seed", "gap_each_side"))
        same_manifests = all((output_dir / f"{name}.csv").exists() and
                             pd.read_csv(output_dir / f"{name}.csv", keep_default_na=False).equals(
                                 part.reset_index(drop=True)) for name, part in splits.items())
        if same_settings and same_manifests:
            return previous  # Reuse frozen splits without rewriting timestamps or files.
        raise ValueError("Existing splits differ. Keep trained splits frozen; choose a new --output-dir.")
    if any((output_dir / f"{name}.csv").exists() for name in splits):
        raise ValueError("Incomplete existing manifests; choose a new --output-dir after auditing them.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(output_dir / f"{name}.csv", index=False)
    excluded.to_csv(output_dir / "excluded_boundary.csv", index=False)
    records.to_csv(output_dir / "cleaned_metadata.csv", index=False)
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--strategy", choices=["auto", "session", "chronological"], default="auto")
    parser.add_argument("--gap", type=int, default=BOUNDARY_GAP)
    arguments = parser.parse_args()
    try:
        selected_csv = find_driving_log(arguments.data_dir, arguments.csv)
        summary = prepare_dataset(selected_csv, arguments.output_dir, arguments.strategy, arguments.gap)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Dataset preparation: {error}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
