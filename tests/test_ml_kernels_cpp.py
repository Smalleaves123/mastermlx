import numpy as np
import pytest

from mastermlx import get_backend, set_backend
from mastermlx.accel.backends import _load_cpp_ml_kernels
from mastermlx.accel.ml_kernels import (
    csr_propagate,
    dbscan_labels,
    dbscan_neighbors,
    gmm_log_gaussian,
    gmm_m_step,
    hmm_backward,
    hmm_forward,
    hmm_viterbi,
    kmeans_assign,
    kmeans_update,
    knn_impute,
    knn_affinity,
    knn_graph,
    rbf_affinity,
    radius_neighbors,
    meanshift_update,
)


def _require_cpp_ml_kernels():
    cpp = _load_cpp_ml_kernels()
    if cpp is None:
        pytest.skip("C++ ML extension is unavailable")
    return cpp


def _require_cpp_radius_neighbors():
    cpp = _require_cpp_ml_kernels()
    if not callable(getattr(cpp, "radius_neighbors", None)):
        pytest.skip("C++ radius-neighbor kernel is unavailable")
    return cpp


def _run_both(func, *args):
    old = get_backend()
    try:
        set_backend("auto")
        _require_cpp_ml_kernels()
        accelerated = func(*args)
        set_backend("numpy")
        fallback = func(*args)
    finally:
        set_backend(old)
    return accelerated, fallback


def _assert_same(accelerated, fallback):
    if isinstance(accelerated, tuple):
        assert len(accelerated) == len(fallback)
        for left, right in zip(accelerated, fallback):
            assert np.allclose(left, right, rtol=1e-10, atol=1e-10)
    else:
        assert np.allclose(accelerated, fallback, rtol=1e-10, atol=1e-10)


def test_ml_kernels_match_numpy_fallback_on_non_contiguous_inputs():
    rng = np.random.default_rng(7)
    X = np.asfortranarray(rng.normal(size=(10, 4)).astype(np.float32))
    centers = np.asfortranarray(rng.normal(size=(3, 4)).astype(np.float32))
    labels = np.arange(10, dtype=np.int64) % 3
    responsibilities = rng.random((10, 3))
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    means = centers.astype(float)
    covariances = np.stack([np.eye(4) * (0.8 + 0.2 * i) for i in range(3)])
    precisions = np.linalg.inv(covariances)
    logdet = np.linalg.slogdet(covariances)[1]
    query_missing = np.array([[np.nan, 0.5, 1.0, 1.5], [2.0, 2.5, np.nan, 3.5]])
    fit_missing = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0]])
    csr_indptr = np.array([0, 2, 3, 5])
    csr_indices = np.array([1, 2, 0, 0, 1])
    csr_weights = np.array([0.4, 0.6, 1.0, 0.25, 0.75])
    radius_query = np.asfortranarray(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32))
    radius_fit = np.asfortranarray(np.array([[0.0, 0.0], [0.6, 0.0], [2.0, 0.0]], dtype=np.float32))
    dbscan_indptr = np.array([0, 2, 4, 6, 8], dtype=np.int64)
    dbscan_indices = np.array([0, 1, 0, 1, 2, 3, 2, 3], dtype=np.int64)
    meanshift_centers = np.asfortranarray(rng.normal(size=(3, 4)).astype(np.float32))
    hmm_seq = np.array([0, 1, 1, 0], dtype=np.int64)
    hmm_start = np.array([0.6, 0.4])
    hmm_trans = np.array([[0.7, 0.3], [0.2, 0.8]])
    hmm_emit = np.array([[0.8, 0.2], [0.3, 0.7]])

    for func, args in (
        (rbf_affinity, (X, 0.7)),
        (knn_affinity, (X, 3)),
        (knn_graph, (X, 3)),
        (radius_neighbors, (radius_query, radius_fit, 0.75)),
        (knn_impute, (query_missing, fit_missing, 2, "distance")),
        (dbscan_neighbors, (X, 2.0)),
        (dbscan_labels, (dbscan_indptr, dbscan_indices, 2)),
        (meanshift_update, (X, meanshift_centers, 1.5)),
        (hmm_forward, (hmm_seq, hmm_start, hmm_trans, hmm_emit)),
        (hmm_backward, (hmm_seq, hmm_start, hmm_trans, hmm_emit)),
        (hmm_viterbi, (hmm_seq, hmm_start, hmm_trans, hmm_emit)),
        (csr_propagate, (csr_indptr, csr_indices, csr_weights, responsibilities[:3])),
        (kmeans_assign, (X, centers)),
        (kmeans_update, (X, labels, 3)),
        (gmm_log_gaussian, (X, means, precisions, logdet)),
        (gmm_m_step, (X, responsibilities, 1e-6)),
    ):
        accelerated, fallback = _run_both(func, *args)
        _assert_same(accelerated, fallback)


def test_ml_kernels_use_numpy_when_backend_is_forced():
    old = get_backend()
    try:
        set_backend("numpy")
        assert _load_cpp_ml_kernels() is None
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
        affinity = rbf_affinity(X, 1.0)
        assert affinity.shape == (3, 3)
        assert np.allclose(np.diag(affinity), 0.0)
    finally:
        set_backend(old)


def test_ml_kernel_validation():
    X = np.ones((3, 2))
    with pytest.raises(ValueError, match="between 1"):
        knn_affinity(X, 3)
    with pytest.raises(ValueError, match="same number of features"):
        kmeans_assign(X, np.ones((2, 3)))
    with pytest.raises(ValueError, match="match the number of samples"):
        kmeans_update(X, np.array([0, 1]), 2)
    with pytest.raises(ValueError, match="finite"):
        rbf_affinity(np.array([[0.0, np.nan]]), 1.0)
    with pytest.raises(ValueError, match="positive and finite"):
        radius_neighbors(X, X, 0.0)
    with pytest.raises(ValueError, match="positive and finite"):
        radius_neighbors(X, X, np.inf)
    with pytest.raises(ValueError, match="positive and finite"):
        meanshift_update(X, X, np.nan)
    old = get_backend()
    try:
        set_backend("numpy")
        with pytest.raises(ValueError, match="non-negative"):
            hmm_forward([0], [1.0, -0.1], [[1.0, 0.0], [0.0, 1.0]], [[1.0], [1.0]])
    finally:
        set_backend(old)


def test_knn_affinity_tie_breaking_matches_compiled_contract():
    _require_cpp_ml_kernels()
    X = np.zeros((6, 2))
    accelerated, fallback = _run_both(knn_affinity, X, 2)
    expected = np.array([
        [0, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
    ], dtype=float)
    assert np.array_equal(accelerated, fallback)
    assert np.array_equal(fallback, expected)


def test_cpp_ml_kernels_validate_direct_inputs():
    old = get_backend()
    try:
        set_backend("auto")
        cpp = _require_cpp_ml_kernels()

        X = np.ones((3, 2))
        with pytest.raises(ValueError, match="2D"):
            cpp.rbf_affinity(np.ones(2), 1.0)
        with pytest.raises(ValueError, match="non-empty"):
            cpp.rbf_affinity(np.empty((0, 2)), 1.0)
        with pytest.raises(ValueError, match="between 1"):
            cpp.knn_affinity(X, 3)
        with pytest.raises(ValueError, match="features"):
            cpp.kmeans_assign(X, np.ones((2, 3)))
        _require_cpp_radius_neighbors()
        with pytest.raises(ValueError, match="positive and finite"):
            cpp.radius_neighbors(X, X, 0.0)
        if callable(getattr(cpp, "dbscan_labels", None)):
            with pytest.raises(ValueError, match="row pointer"):
                cpp.dbscan_labels(np.array([0, 2, 1], dtype=np.int64), np.array([0, 1], dtype=np.int64), 1)
    finally:
        set_backend(old)


def test_radius_neighbors_returns_stable_csr_rows():
    _require_cpp_radius_neighbors()
    X = np.asfortranarray(np.array([[0.0], [1.0], [2.0]]))
    X_fit = np.asfortranarray(np.array([[0.0], [0.5], [1.5], [3.0]]))
    accelerated, fallback = _run_both(radius_neighbors, X, X_fit, 0.51)

    for result in (accelerated, fallback):
        indptr, indices, distances = result
        assert indptr.dtype == np.int64
        assert indices.dtype == np.int64
        assert np.array_equal(indptr, np.array([0, 2, 4, 5]))
        assert np.array_equal(indices, np.array([0, 1, 1, 2, 2]))
        assert np.allclose(distances, np.array([0.0, 0.5, 0.5, 0.5, 0.5]))


def test_radius_neighbors_numpy_backend_does_not_require_cpp_radius_api(monkeypatch):
    import mastermlx.accel.ml_kernels as kernels

    old = get_backend()
    try:
        set_backend("numpy")
        monkeypatch.setattr(kernels, "_load_cpp_ml_kernels", lambda: object())
        indptr, indices, distances = kernels.radius_neighbors(
            np.array([[0.0], [2.0]]), np.array([[0.0], [1.0], [3.0]]), 1.01
        )
    finally:
        set_backend(old)

    assert np.array_equal(indptr, np.array([0, 2, 4]))
    assert np.array_equal(indices, np.array([0, 1, 1, 2]))
    assert np.allclose(distances, np.array([0.0, 1.0, 1.0, 1.0]))


def test_dbscan_labels_numpy_fallback_handles_old_cpp_extension(monkeypatch):
    import mastermlx.accel.ml_kernels as kernels

    old = get_backend()
    try:
        set_backend("auto")
        monkeypatch.setattr(kernels, "_load_cpp_ml_kernels", lambda: object())
        labels, core_samples = kernels.dbscan_labels(
            np.array([0, 2, 4, 6], dtype=np.int64),
            np.array([0, 1, 0, 1, 2, 2], dtype=np.int64),
            2,
        )
    finally:
        set_backend(old)

    assert np.array_equal(labels, np.array([0, 0, 1]))
    assert np.array_equal(core_samples, np.array([0, 1, 2]))


def test_meanshift_numpy_fallback_handles_old_cpp_extension(monkeypatch):
    import mastermlx.accel.ml_kernels as kernels

    old = get_backend()
    try:
        set_backend("auto")
        monkeypatch.setattr(kernels, "_load_cpp_ml_kernels", lambda: object())
        updates = kernels.meanshift_update(
            np.array([[0.0], [0.2], [2.0]]), np.array([[0.1], [2.0]]), 0.25
        )
    finally:
        set_backend(old)

    assert np.allclose(updates, np.array([[0.1], [2.0]]))
