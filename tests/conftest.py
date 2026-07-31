import numpy as np
import pytest


@pytest.fixture
def linear_binary_data():
    """Small deterministic binary dataset shared by estimator tests."""

    X = np.array([[-2.0, 0.0], [-1.0, 1.0], [1.0, -1.0], [2.0, 0.0]])
    y = np.array([0, 0, 1, 1])
    return X, y
