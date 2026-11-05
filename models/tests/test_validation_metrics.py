"""Cosine and error metrics match the functional-sim validation convention."""
import numpy as np
import pytest

from validation_metrics import compare_cosine_max_mean


def test_cosine_max_mean_for_perturbed_tensor():
    actual = np.array([1.0, 2.0, 3.0])
    expected = np.array([1.0, 2.0, 2.0])

    cosine, max_error, mean_error = compare_cosine_max_mean(actual, expected)

    assert cosine == pytest.approx(11 / (np.sqrt(14) * 3))
    assert max_error == pytest.approx(1.0)
    assert mean_error == pytest.approx(1 / 3)


def test_cosine_is_zero_when_either_tensor_has_zero_norm():
    cosine, max_error, mean_error = compare_cosine_max_mean(
        np.zeros(3), np.zeros(3)
    )

    # Match functional-sim core/kernel_ops.py semantics.
    assert cosine == 0.0
    assert max_error == 0.0
    assert mean_error == 0.0


def test_metric_rejects_different_tensor_shapes():
    with pytest.raises(ValueError, match="same shape"):
        compare_cosine_max_mean(np.zeros((2, 2)), np.zeros(4))
