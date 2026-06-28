import numpy as np

from mastermlx.clustering import AgglomerativeClustering


def test_agglomerative_clustering_finds_two_groups():
    X = np.array([
        [0.0, 0.0],
        [0.1, -0.1],
        [-0.1, 0.1],
        [4.9, 5.0],
        [5.0, 5.1],
        [5.1, 4.9],
    ])

    model = AgglomerativeClustering(n_clusters=2, linkage="average")
    labels = model.fit_predict(X)

    assert labels.shape == (6,)
    assert model.cluster_centers_.shape == (2, 2)
    assert model.children_.shape == (4, 2)
    assert set(labels.tolist()) <= {0, 1}
