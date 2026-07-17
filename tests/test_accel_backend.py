import numpy as np
import pytest

from mastermlx import get_backend, set_backend
from mastermlx.accel import (
    get_active_backend,
    pairwise_distances,
    pairwise_manhattan_distances,
    pairwise_squared_euclidean,
)
from mastermlx.accel.backends import (
    _load_cpp_backend,
    _load_cpp_kernels,
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


def test_cpp_extensions_validate_direct_inputs():
    cpp = _load_cpp_backend()
    kernels = _load_cpp_kernels()
    if cpp is None or kernels is None:
        pytest.skip("C++ extensions are unavailable")

    X = np.ones((1, 2))
    with pytest.raises(ValueError, match="2D"):
        cpp.pairwise_distances(np.ones(2), X)
    with pytest.raises(ValueError, match="non-empty"):
        cpp.pairwise_distances(np.empty((0, 2)), X)
    with pytest.raises(ValueError, match="same number of features"):
        cpp.pairwise_distances(X, np.ones((1, 3)))
    with pytest.raises(ValueError, match="positive"):
        cpp.pairwise_minkowski(X, X, 0.0)
    with pytest.raises(ValueError, match="finite"):
        kernels.rbf_kernel(X, X, np.nan)
    with pytest.raises(ValueError, match="match"):
        kernels.rbf_kernel_fast(X, X, np.ones(2), np.ones(1), 1.0)

    non_contiguous = np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3, 4))
    result = cpp.pairwise_squared_euclidean(non_contiguous, non_contiguous)
    assert result.shape == (3, 3)
    assert result.dtype == np.float64
