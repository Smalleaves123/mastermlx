import numpy as np

from mastermlx.decomposition import KernelPCA
from mastermlx.neighbors import KNNClassifier
from mastermlx.utils import (
    chebyshev_distance,
    cosine_distance,
    hamming_distance,
    jaccard_distance,
    mahalanobis_distance,
    minkowski_distance,
    pairwise_distance,
)
from mastermlx.math_tools import additive_chi2_kernel, cosine_kernel
from mastermlx.utils.kernels import pairwise_kernel


def test_distance_metrics_cover_common_cases():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 0.0, 4.0])

    assert np.isclose(minkowski_distance(a, b, p=2), np.sqrt(5.0))
    assert np.isclose(chebyshev_distance(a, b), 2.0)
    assert np.isclose(cosine_distance(a, b), 1.0 - (13.0 / (np.sqrt(14.0) * np.sqrt(17.0))))
    assert np.isclose(hamming_distance(a, b), 2.0 / 3.0)
    assert np.isclose(jaccard_distance(a > 0, b > 0), 1.0 / 3.0)
    assert np.isclose(mahalanobis_distance(a, b), np.sqrt(5.0))


def test_pairwise_distance_supports_new_metrics():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    Y = np.array([[1.0, 1.0]])

    D = pairwise_distance(X, Y, metric="cosine")
    H = pairwise_distance(X, Y, metric="hamming")
    J = pairwise_distance(X > 0, Y > 0, metric="jaccard")
    M = pairwise_distance(X, Y, metric="mahalanobis")

    assert D.shape == (2, 1)
    assert np.isfinite(D).all()
    assert np.isclose(D[0, 0], 1.0 - 1.0 / np.sqrt(2.0))
    assert H.shape == (2, 1)
    assert J.shape == (2, 1)
    assert M.shape == (2, 1)
    assert np.isfinite(H).all()
    assert np.isfinite(J).all()
    assert np.isfinite(M).all()
    assert np.isclose(H[0, 0], 0.5)
    assert np.isclose(J[0, 0], 0.5)
    assert np.isclose(M[0, 0], 1.0)


def test_neighbor_models_accept_extended_metrics():
    X = np.array([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.9, 0.1],
    ])
    y = np.array([0, 0, 1, 1])

    model = KNNClassifier(k=1, metric="cosine").fit(X, y)

    assert model.predict([1.0, 0.0, 0.0]) == 0
    assert model.predict([0.0, 1.0, 0.0]) == 1


def test_pairwise_kernels_cover_extended_family():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    Y = np.array([[1.0, 1.0]])

    for kernel in ("cosine", "laplacian", "sigmoid", "chi2", "additive_chi2", "hellinger"):
        K = pairwise_kernel(X, Y, kernel=kernel, gamma=1.0, coef0=0.5)
        assert K.shape == (2, 1)
        assert np.isfinite(K).all()

    assert cosine_kernel(X, Y).shape == (2, 1)
    assert additive_chi2_kernel(X + 1.0, Y + 1.0).shape == (2, 1)


def test_kernel_validation_and_hellinger_stability():
    X = np.array([[1e200]])
    Y = np.array([[1e200]])

    hellinger = pairwise_kernel(X, Y, kernel="hellinger")
    assert hellinger.shape == (1, 1)
    assert np.isfinite(hellinger).all()
    assert np.isclose(hellinger[0, 0], 1e200)

    bad = np.array([[-1.0, 0.5]])
    with np.testing.assert_raises(ValueError):
        pairwise_kernel(bad, np.array([[1.0, 0.5]]), kernel="chi2")
    with np.testing.assert_raises(ValueError):
        pairwise_kernel(bad, np.array([[1.0, 0.5]]), kernel="additive_chi2")
    with np.testing.assert_raises(ValueError):
        pairwise_kernel(bad, np.array([[1.0, 0.5]]), kernel="hellinger")


def test_minkowski_pairwise_rejects_nonpositive_p():
    X = np.array([[0.0, 0.0]])
    Y = np.array([[1.0, 1.0]])

    with np.testing.assert_raises(ValueError):
        pairwise_distance(X, Y, metric="minkowski", p=0)
    with np.testing.assert_raises(ValueError):
        pairwise_distance(X, Y, metric="minkowski", p=-1)


def test_kernel_pca_supports_laplacian_kernel():
    X = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])

    model = KernelPCA(n_components=2, kernel="laplacian", gamma=1.0).fit(X)
    Z = model.transform(X)

    assert Z.shape == (4, 2)
    assert np.isfinite(Z).all()
