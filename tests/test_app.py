"""Smoke tests for the no-data screen and fixture-based dashboard interaction."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import config

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_dashboard_without_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception
    assert len(app.metric) == 0
    assert any("Select a local driving log" in element.value for element in app.warning)
    assert "Self-Driving Car Behavioral Cloning" in app.title[0].value


def test_dashboard_with_fixture_and_image_selection(make_dataset, monkeypatch):
    csv_path = make_dataset(count=12)
    monkeypatch.setattr(config, "DATA_DIR", csv_path.parent)
    monkeypatch.setattr(config, "MODELS_DIR", csv_path.parent / "models")
    monkeypatch.setattr(config, "REPORTS_DIR", csv_path.parent / "reports")
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception
    assert any(metric.label == "Valid records" and metric.value == "12" for metric in app.metric)
    app.number_input[0].set_value(5).run()
    assert not app.exception
    assert any("Source row: 5" in caption.value for caption in app.caption)
    assert any("(80, 160, 3)" in code.value for code in app.code)
    assert len(app.tabs) == 6
    assert any("Trained model not found" in item.value for item in app.warning)
