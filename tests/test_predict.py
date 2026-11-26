import io

import numpy as np
import pytest
from PIL import Image

tf = pytest.importorskip("tensorflow")

from src.model import build_model
from src.pipeline import file_sha256, preprocessing_settings, write_json
from src.predict import load_trained_model, predict_steering
from src.preprocess import preprocess_image


@pytest.fixture
def saved_model(tmp_path):
    # Untrained unit-test fixture only: never copied to project models or reports.
    model = build_model()
    path = tmp_path / "best_model.keras"
    model.save(path)
    settings = preprocessing_settings()
    settings.update(crop_top=0.2, crop_bottom=0.15)
    write_json(tmp_path / "model_metadata.json", {
        "training_status": "complete", "model_sha256": file_sha256(path), "preprocessing": settings})
    return path, settings


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_reload_predict_uses_saved_crop_without_double_normalization(saved_model, mode):
    path, settings = saved_model
    model, metadata = load_trained_model(path)
    image = Image.new(mode, (320, 160), 128)
    actual = predict_steering(image, model)
    pixels = preprocess_image(image, settings["crop_top"], settings["crop_bottom"])
    expected = float(np.asarray(model(pixels[None], training=False))[0, 0])
    assert isinstance(actual, float) and np.isfinite(actual)
    assert actual == pytest.approx(expected)
    assert metadata["preprocessing"] == settings


def test_uploaded_bytes_and_invalid_image(saved_model):
    model, _ = load_trained_model(saved_model[0])
    upload = io.BytesIO()
    Image.new("RGB", (320, 160), "green").save(upload, format="PNG")
    upload.seek(0)
    assert np.isfinite(predict_steering(upload, model))
    with pytest.raises(OSError):
        predict_steering(io.BytesIO(b"not an image"), model)


def test_missing_model_and_mismatched_metadata(tmp_path, saved_model):
    with pytest.raises(FileNotFoundError, match="Trained model not found"):
        load_trained_model(tmp_path / "absent.keras")
    path, _ = saved_model
    with path.open("ab") as output:
        output.write(b"modified")
    with pytest.raises(ValueError, match="do not match"):
        load_trained_model(path)


@pytest.mark.parametrize("output", [np.array([[np.nan]]), np.array([[1, 2]]), np.array([[np.inf]])])
def test_invalid_model_output_rejected(output):
    with pytest.raises(ValueError, match="finite steering"):
        predict_steering(Image.new("RGB", (320, 160)), lambda pixels, training: output, preprocessing_settings())
