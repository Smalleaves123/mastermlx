import numpy as np

from mastermlx.clustering import KMeans


def test_kmeans_clusters_simple_blobs():
    X = np.array([
        [0.0, 0.0],
        [0.1, -0.1],
        [-0.1, 0.1],
        [5.0, 5.0],
        [5.1, 4.9],
        [4.9, 5.1],
    ])

    model = KMeans(n_clusters=2, random_state=0, n_init=5)
    labels = model.fit_predict(X)

    assert labels.shape == (6,)
    assert set(labels.tolist()) <= {0, 1}
    assert model.cluster_centers_.shape == (2, 2)
    assert model.inertia_ >= 0.0


def test_kmeans_transform_returns_distances():
    X = np.array([
        [0.0, 0.0],
        [10.0, 10.0],
    ])

    model = KMeans(n_clusters=2, random_state=0, n_init=1).fit(X)
    D = model.transform(X)

    assert D.shape == (2, 2)
    assert np.allclose(np.min(D, axis=1), 0.0, atol=1e-8)
