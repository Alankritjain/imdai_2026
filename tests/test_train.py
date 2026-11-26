from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

tf = pytest.importorskip("tensorflow")

import config
from src.evaluate import evaluate_model
from src.predict import load_trained_model, predict_steering
from src.prepare_data import prepare_dataset
from src.train import train_model


def test_training_evaluation_inference_and_streamlit(make_dataset, tmp_path, monkeypatch):
    """End-to-end SYNTHETIC code test: all generated outputs stay in pytest's folder."""
    csv_path = make_dataset(count=40)
    processed, models, reports = (tmp_path / name for name in ("processed", "models", "reports"))
    prepare_dataset(csv_path, processed, gap=1)
    metadata = train_model(processed, models, reports, epochs=2, batch_size=8)
    assert metadata["epochs_completed"] == 2
    assert np.isfinite(metadata["best_validation_loss"])
    history = pd.read_csv(reports / "training_history.csv")
    assert np.isfinite(history[["loss", "val_loss"]]).all().all()
    model, saved_metadata = load_trained_model(models / "best_model.keras")
    test = pd.read_csv(processed / "test.csv")
    assert np.isfinite(predict_steering(test.iloc[0].image_path, model))
    metrics = evaluate_model(processed, models / "best_model.keras", reports, batch_size=8)
    assert metrics["test_records"] == len(test)
    training_mean = pd.read_csv(processed / "train.csv").steering.mean()
    assert metrics["baseline_training_mean"] == pytest.approx(training_mean)
    assert metrics["baseline"]["mae"] == pytest.approx(np.mean(np.abs(test.steering - training_mean)))
    predictions = pd.read_csv(reports / "predictions.csv")
    assert len(predictions) == len(test)
    assert predictions.predicted_steering.iloc[0] == pytest.approx(predict_steering(test.iloc[0].image_path, model))
    for filename in ("training_loss.png", "actual_vs_predicted.png", "error_distribution.png", "sample_predictions.png"):
        assert (reports / "figures" / filename).stat().st_size > 0
    for key, value in {"DATA_DIR": tmp_path, "MODELS_DIR": models, "REPORTS_DIR": reports,
                       "PROCESSED_DIR": processed}.items():
        monkeypatch.setattr(config, key, value)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert any(metric.label == "CNN MAE" for metric in app.metric)
    # A real prediction now appears immediately for the first built-in held-out example.
    assert any(metric.label == "Predicted Steering" and np.isfinite(float(metric.value)) for metric in app.metric)
    assert not app.error
    image_path = Path(test.iloc[0].image_path)
    app.radio(key="prediction_source").set_value("Upload image").run()
    app.file_uploader[0].set_value((image_path.name, image_path.read_bytes(), "image/jpeg")).run()
    assert not app.exception and not app.error
    uploaded_prediction = next(metric for metric in app.metric if metric.label == "Predicted Steering")
    assert float(uploaded_prediction.value) == pytest.approx(
        predict_steering(image_path, model), abs=1e-6)
    app.file_uploader[0].set_value(("broken.jpg", b"invalid JPEG", "image/jpeg")).run()
    assert not app.exception
    assert any("Invalid image" in error.value for error in app.error)
    app.radio(key="prediction_source").set_value("Recorded replay").run()
    assert not app.exception
    assert any("Recorded frame 1" in item.value for item in app.caption)
    assert any(metric.label == "Recorded steering" for metric in app.metric)
    with pytest.raises(FileExistsError):
        train_model(processed, models, reports, epochs=1)
    test.loc[0, "steering"] += 0.1
    test.to_csv(processed / "test.csv", index=False)
    with pytest.raises(ValueError, match="changed since training"):
        evaluate_model(processed, models / "best_model.keras", reports)
