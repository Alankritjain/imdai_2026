"""Artificial fixtures used only by tests, never by the demonstration reports."""

import csv

import pytest
from PIL import Image


@pytest.fixture
def make_dataset(tmp_path):
    def create(rows=None, header=("center", "steering"), count=100):
        image_dir = tmp_path / "IMG"
        image_dir.mkdir(exist_ok=True)
        for index in range(count):
            # Separate colors widely so JPEG quantization cannot make adjacent fixtures identical.
            color = (index * 37 % 256, index * 73 % 256, index * 109 % 256)
            Image.new("RGB", (320, 160), color).save(image_dir / f"center_{index}.jpg")
        if rows is None:
            rows = [[f"IMG/center_{index}.jpg", (index % 11 - 5) / 10] for index in range(count)]
        csv_path = tmp_path / "driving_log.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            if header:
                writer.writerow(header)
            writer.writerows(rows)
        return csv_path
    return create
