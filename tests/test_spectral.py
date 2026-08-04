import numpy as np

from mastermlx.clustering import SpectralClustering


def test_spectral_clustering_finds_two_clusters():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [4.9, 5.0],
        [5.0, 5.1],
        [5.1, 4.9],
    ])

    model = SpectralClustering(n_clusters=2, affinity="rbf", gamma=1.0, random_state=0)
    labels = model.fit_predict(X)

    assert labels.shape == (6,)
    assert model.embedding_.shape == (6, 2)
    assert model.affinity_matrix_.shape == (6, 6)
    assert set(labels.tolist()) <= {0, 1}


def test_spectral_nearest_neighbor_affinity_is_symmetric_binary():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [4.9, 5.0],
        [5.0, 5.1],
        [5.1, 4.9],
    ])

    model = SpectralClustering(n_clusters=2, affinity="nearest_neighbors", n_neighbors=2)
    W = model._build_affinity(X)

    assert np.array_equal(W, W.T)
    assert np.all((W == 0.0) | (W == 1.0))
    assert np.all(np.diag(W) == 0.0)
