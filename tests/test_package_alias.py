import numpy as np

import mastermlx
from mastermlx.clustering import KMeans


def test_mastermlx_package_exports_backend_controls():
    assert hasattr(mastermlx, "get_backend")
    assert hasattr(mastermlx, "set_backend")
    assert hasattr(mastermlx, "BernoulliNB")
    assert hasattr(mastermlx, "MultinomialNB")
    assert hasattr(mastermlx, "precision_score")
    assert hasattr(mastermlx, "autocorrelation")


def test_mastermlx_submodule_imports_work():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    model = KMeans(n_clusters=2, n_init=1, random_state=0).fit(X)
    assert model.cluster_centers_.shape == (2, 2)
