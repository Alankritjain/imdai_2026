import numpy as np
import pytest

from src.evaluate import regression_metrics


def test_metrics_known_values():
    actual, predicted = [0, 1, 2], [0, 0, 4]
    metrics = regression_metrics(actual, predicted)
    assert metrics["mae"] == 1
    assert metrics["rmse"] == pytest.approx(np.sqrt(5 / 3))


@pytest.mark.parametrize("actual,predicted", [([], []), ([1], [1, 2]), ([1], [np.nan]), ([[1]], [[1]])])
def test_bad_metrics_inputs(actual, predicted):
    with pytest.raises(ValueError):
        regression_metrics(actual, predicted)
