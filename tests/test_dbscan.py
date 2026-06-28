import numpy as np

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
