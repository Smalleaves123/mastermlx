import numpy as np

from mastermlx.decomposition import NMF


def test_nmf_factorizes_non_negative_matrix():
    X = np.array([
        [1.0, 0.5, 0.0],
        [0.8, 0.4, 0.1],
        [0.0, 0.2, 1.0],
        [0.1, 0.3, 0.9],
    ])

    nmf = NMF(n_components=2, max_iter=1000, tol=1e-6, random_state=0)
    W = nmf.fit_transform(X)
    X2 = nmf.inverse_transform(W)

    assert W.shape == (4, 2)
    assert nmf.components_.shape == (2, 3)
    assert X2.shape == X.shape
    assert nmf.reconstruction_err_ >= 0.0
