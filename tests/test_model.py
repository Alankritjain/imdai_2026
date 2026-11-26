import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.model import build_model


def test_milestone_architecture_compilation_and_forward_pass():
    model = build_model(0.002)
    assert model.input_shape == (None, 80, 160, 3)
    assert model.output_shape == (None, 1)
    assert model.count_params() == 21329
    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)
    assert float(model.optimizer.learning_rate.numpy()) == pytest.approx(0.002)
    assert model.loss == "mse"
    prediction = np.asarray(model(np.zeros((2, 80, 160, 3), np.float32), training=False))
    assert prediction.shape == (2, 1)
    assert np.isfinite(prediction).all()
