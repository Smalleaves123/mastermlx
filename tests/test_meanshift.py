import numpy as np
import pytest

from mastermlx import get_backend, set_backend
from mastermlx.clustering import MeanShift


def test_meanshift_finds_dense_modes():
    X = np.array([
        [0.0, 0.0],
        [0.1, -0.1],
        [-0.1, 0.1],
        [5.0, 5.0],
        [5.1, 5.0],
        [5.0, 5.1],
    ])

    model = MeanShift(bandwidth=0.5)
    labels = model.fit_predict(X)

    assert labels.shape == (6,)
    assert model.cluster_centers_.shape[1] == 2
    assert model.n_clusters_ >= 2
    assert set(labels.tolist()) <= set(range(model.n_clusters_))


def test_meanshift_rejects_invalid_iteration_parameters():
    X = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError, match="max_iter"):
        MeanShift(max_iter=0).fit(X)
    with pytest.raises(ValueError, match="non-negative"):
        MeanShift(tol=-1.0).fit(X)


def test_meanshift_cpp_and_numpy_match_for_integer_inputs():
    X = np.array([[0, 0], [1, 0], [0, 1], [5, 5], [6, 5], [5, 6]])
    old = get_backend()
    try:
        set_backend("auto")
        accelerated = MeanShift(bandwidth=1.5, max_iter=20, tol=1e-5).fit(X)
        set_backend("numpy")
        fallback = MeanShift(bandwidth=1.5, max_iter=20, tol=1e-5).fit(X)
    finally:
        set_backend(old)

    assert np.allclose(accelerated.cluster_centers_, fallback.cluster_centers_)
    assert np.array_equal(accelerated.labels_, fallback.labels_)
    assert accelerated.n_iter_ == fallback.n_iter_
