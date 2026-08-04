import numpy as np
import pytest

from mastermlx import get_backend, set_backend
from mastermlx.clustering import DBSCAN


def test_dbscan_finds_two_dense_clusters():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [5.0, 5.0],
        [5.1, 5.0],
        [5.0, 5.1],
        [10.0, 10.0],
    ])

    model = DBSCAN(eps=0.25, min_samples=2)
    labels = model.fit_predict(X)

    assert labels.shape == (7,)
    assert model.n_clusters_ == 2
    assert labels[-1] == -1
    assert set(labels.tolist()) <= {-1, 0, 1}


def test_dbscan_cpp_and_numpy_label_expansion_match():
    rng = np.random.default_rng(9)
    X = np.vstack((rng.normal(0.0, 0.05, size=(24, 2)), rng.normal(3.0, 0.05, size=(24, 2))))
    X = np.asfortranarray(X)
    old = get_backend()
    try:
        set_backend("auto")
        accelerated = DBSCAN(eps=0.2, min_samples=3).fit(X)
        set_backend("numpy")
        fallback = DBSCAN(eps=0.2, min_samples=3).fit(X)
    finally:
        set_backend(old)

    assert np.array_equal(accelerated.labels_, fallback.labels_)
    assert np.array_equal(accelerated.core_sample_indices_, fallback.core_sample_indices_)
    assert accelerated.n_clusters_ == fallback.n_clusters_ == 2


@pytest.mark.parametrize("eps", [np.inf, np.nan])
def test_dbscan_rejects_nonfinite_eps(eps):
    with pytest.raises(ValueError, match="positive and finite"):
        DBSCAN(eps=eps).fit(np.array([[0.0], [1.0]]))
