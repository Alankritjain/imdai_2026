"""Download a small, genuine contiguous subset for the lab (no Kaggle credentials).

This optional utility uses Kaggle's public file endpoint. If that endpoint changes
or requires login, use the manual download instructions in README.md instead.
Only center-camera images are downloaded; original steering labels are unchanged.
"""

import csv
import hashlib
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_FILE = "dataset/dataset/driving_log.csv"
BASE_URL = "https://www.kaggle.com/api/v1/datasets/download/zaynena/selfdriving-car-simulator/"
START_ROW = 0
ROW_COUNT = 300


def download_file(remote_path):
    request = Request(BASE_URL + quote(remote_path, safe=""), headers={"User-Agent": "behavioral-cloning-lab/1.0"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=45) as response:
                content = response.read()
            if zipfile.is_zipfile(io.BytesIO(content)):
                with zipfile.ZipFile(io.BytesIO(content)) as archive:
                    names = [name for name in archive.namelist() if not name.endswith("/")]
                    if len(names) != 1:
                        raise ValueError("Expected one file from the Kaggle file endpoint.")
                    content = archive.read(names[0])
            return content
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1)


def main():
    source_dir = PROJECT_DIR / "data" / "raw" / "source_metadata"
    subset_dir = PROJECT_DIR / "data" / "raw" / "lab_subset"
    source_dir.mkdir(parents=True, exist_ok=True)
    (subset_dir / "IMG").mkdir(parents=True, exist_ok=True)
    source_path = source_dir / "original_log.source.csv"
    if not source_path.exists():
        source_path.write_bytes(download_file(SOURCE_FILE))
    source_bytes = source_path.read_bytes()
    rows = list(csv.reader(io.StringIO(source_bytes.decode("utf-8-sig"))))
    selected_rows = rows[START_ROW:START_ROW + ROW_COUNT]
    if len(selected_rows) != ROW_COUNT or any(len(row) != 7 for row in selected_rows):
        raise ValueError("Source format changed; inspect it before creating a subset.")

    def download_center(row):
        filename = row[0].strip().replace("\\", "/").split("/")[-1]
        destination = subset_dir / "IMG" / filename
        if not destination.exists():
            content = download_file("dataset/dataset/IMG/" + filename)
            if not content.startswith(b"\xff\xd8"):
                raise ValueError(f"Expected JPEG data for {filename}; use manual Kaggle download.")
            destination.write_bytes(content)
        return filename

    # A few simultaneous network requests keep this optional download practical.
    with ThreadPoolExecutor(max_workers=8) as executor:
        for count, filename in enumerate(executor.map(download_center, selected_rows), start=1):
            if count % 25 == 0:
                print(f"Downloaded/checked {count}/{ROW_COUNT} center images", flush=True)
    csv_path = subset_dir / "driving_log.csv"
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as existing:
            if list(csv.reader(existing)) != selected_rows:
                raise ValueError("Existing lab driving_log.csv differs; select another output directory.")
    else:
        with csv_path.open("w", newline="", encoding="utf-8") as output:
            csv.writer(output).writerows(selected_rows)
    provenance = {
        "dataset_source": "zaynena/selfdriving-car-simulator", "source_file": SOURCE_FILE,
        "source_url": "https://www.kaggle.com/datasets/zaynena/selfdriving-car-simulator",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_log_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_log_records": len(rows), "subset_start_row_zero_based": START_ROW,
        "subset_record_count": len(selected_rows), "analysis_scope": "subset",
        "selection_method": "First 300 consecutive source rows; no steering-based selection.",
        "note": "Original rows retained. Only center images downloaded. Not representative of all tracks.",
    }
    (subset_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Real lab subset ready: {csv_path}")


if __name__ == "__main__":
    main()
