import numpy as np

from mastermlx.decomposition import PCA


def test_pca_reduces_dimension():
    X = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
    ])

    pca = PCA(n_components=1)
    Z = pca.fit_transform(X)

    assert Z.shape == (4, 1)
    assert pca.components_.shape == (1, 2)


def test_pca_inverse_transform_shape():
    X = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
    ])

    pca = PCA(n_components=1).fit(X)
    Z = pca.transform(X)
    X2 = pca.inverse_transform(Z)

    assert X2.shape == X.shape

