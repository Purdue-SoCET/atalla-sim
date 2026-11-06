"""Numerical metrics used by the model-runner kernel validation suite."""
from __future__ import annotations

import numpy as np


def compare_cosine_max_mean(actual, expected) -> tuple[float, float, float]:
    """Return cosine similarity, max absolute error, and mean absolute error.

    The zero-norm convention matches the functional simulator's validation
    helper: cosine similarity is 0 when either vector has zero norm.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(
            f"tensors must have the same shape, got {actual.shape} and {expected.shape}"
        )
    if actual.size == 0:
        raise ValueError("cannot compare empty tensors")
    a = actual.ravel().astype(np.float64)
    e = expected.ravel().astype(np.float64)
    norm_a = np.linalg.norm(a)
    norm_e = np.linalg.norm(e)
    cosine = float(np.dot(a, e) / (norm_a * norm_e)) if norm_a > 0 and norm_e > 0 else 0.0
    difference = np.abs(a - e)
    return cosine, float(np.max(difference)), float(np.mean(difference))
