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


def test_pca_tall_matrix_matches_svd_statistics():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 10))

    model = PCA(n_components=3).fit(X)
    _, singular_values, vectors = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)

    np.testing.assert_allclose(
        model.explained_variance_, singular_values[:3] ** 2 / (X.shape[0] - 1), rtol=1e-10
    )
    np.testing.assert_allclose(np.abs(model.components_), np.abs(vectors[:3]), rtol=1e-8)
