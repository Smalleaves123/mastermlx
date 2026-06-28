import numpy as np

from mastermlx.clustering import AffinityPropagation


def test_affinity_propagation_identifies_exemplars():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [5.0, 5.0],
        [5.1, 5.0],
        [5.0, 5.1],
    ])

    model = AffinityPropagation(damping=0.7, max_iter=200, convergence_iter=10, preference=None)
    labels = model.fit_predict(X)

    assert labels.shape == (6,)
    assert model.cluster_centers_.shape[1] == 2
    assert model.cluster_centers_indices_.ndim == 1
    assert model.n_clusters_ >= 1
