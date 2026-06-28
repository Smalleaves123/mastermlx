import numpy as np

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
