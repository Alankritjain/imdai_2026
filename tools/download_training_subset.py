"""Download a larger chronological excerpt without touching the original lab data."""

import argparse
import csv
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from tools.download_demo_subset import PROJECT_DIR, SOURCE_FILE, download_file


def download_subset(start=1000, count=2400, stride=4, workers=8):
    source_path = PROJECT_DIR / "data/raw/source_metadata/original_log.source.csv"
    source_bytes = source_path.read_bytes()
    rows = list(csv.reader(io.StringIO(source_bytes.decode("utf-8-sig"))))
    selected = rows[start:start + count * stride:stride]
    if len(selected) != count:
        raise ValueError("The requested interval exceeds the source log.")
    names = [row[0].strip().replace("\\", "/").split("/")[-1] for row in selected]
    if names != sorted(names) or len(set(names)) != count:
        raise ValueError("This interval crosses a recording boundary or contains duplicate frames.")
    destination = PROJECT_DIR / "data/raw/training_subset"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "IMG").mkdir(exist_ok=True)
    log_path = destination / "driving_log.csv"
    if log_path.exists():
        with log_path.open(newline="", encoding="utf-8") as source:
            if list(csv.reader(source)) != selected:
                raise ValueError("Existing subset differs. Preserve it and choose another output directory.")

    def download_image(name):
        path = destination / "IMG" / name
        if not path.exists():
            content = download_file("dataset/dataset/IMG/" + name)
            with Image.open(io.BytesIO(content)) as image:
                image.verify()
            path.write_bytes(content)
        with Image.open(path) as image:
            image.convert("RGB").load()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, _ in enumerate(executor.map(download_image, names), start=1):
            if index % 100 == 0 or index == count:
                print(f"Verified {index}/{count} real center images", flush=True)
    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as output:
            csv.writer(output).writerows(selected)
    provenance = {
        "dataset_source": "zaynena/selfdriving-car-simulator", "source_file": SOURCE_FILE,
        "source_log_sha256": hashlib.sha256(source_bytes).hexdigest(), "source_log_records": len(rows),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "analysis_scope": "subset",
        "subset_start_row_zero_based": start, "subset_record_count": count, "source_row_stride": stride,
        "selection_method": "Fixed chronological interval sampled uniformly; selection independent of steering labels.",
        "first_image": names[0], "last_image": names[-1],
        "note": "One recording, not independent drives. Original 300-frame experiment remains separate."
    }
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Completed real dataset: {log_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1000)
    parser.add_argument("--count", type=int, default=2400)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    download_subset(**vars(parser.parse_args()))
