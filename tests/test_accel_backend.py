import numpy as np

from mastermlx import get_backend, set_backend
from mastermlx.accel import (
    get_active_backend,
    pairwise_distances,
    pairwise_manhattan_distances,
    pairwise_squared_euclidean,
)
from mastermlx.accel.backends import (
    _load_cpp_backend,
    _load_cython_backend,
    _numpy_pairwise_squared_euclidean,
)


def test_backend_switching_and_pairwise_ops():
    old = get_backend()
    try:
        set_backend("numpy")
        assert get_active_backend() == "numpy"
        assert _load_cpp_backend() is None
        assert _load_cython_backend() is None

        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        Y = np.array([[1.0, 0.0], [2.0, 2.0]])

        sq = pairwise_squared_euclidean(X, Y)
        eu = pairwise_distances(X, Y)
        man = pairwise_manhattan_distances(X, Y)

        assert sq.shape == (2, 2)
        assert eu.shape == (2, 2)
        assert man.shape == (2, 2)
        assert np.allclose(eu ** 2, sq)
    finally:
        set_backend(old)


def test_numpy_squared_distance_fallback_matches_reference():
    X = np.array([[0.0, 1.0, 2.0], [2.0, -1.0, 0.5]])
    Y = np.array([[1.0, 1.0, 1.0], [-2.0, 0.0, 3.0]])
    reference = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)

    assert np.allclose(_numpy_pairwise_squared_euclidean(X, Y), reference)


def test_backend_report_includes_neural_and_signal_capabilities():
    old = get_backend()
    try:
        set_backend("numpy")
        report = __import__("mastermlx.accel", fromlist=["backend_report"]).backend_report()
        for name in ("cnn", "conv1d", "rnn", "signal"):
            assert name in report
            assert report[name] is False
    finally:
        set_backend(old)
