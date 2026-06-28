import numpy as np

from mastermlx.decomposition import KernelPCA, PCA


def test_kernel_pca_linear_matches_pca_direction():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.2],
            [2.0, 1.9],
            [3.0, 3.1],
            [4.0, 4.2],
        ]
    )

    pca = PCA(n_components=1).fit(X)
    kpca = KernelPCA(n_components=1, kernel="linear").fit(X)

    a = pca.transform(X).ravel()
    b = kpca.transform(X).ravel()
    corr = np.corrcoef(a, b)[0, 1]

    assert abs(corr) > 0.99


def test_kernel_pca_rbf_has_centered_embedding():
    X = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
    )

    model = KernelPCA(n_components=2, kernel="rbf", gamma=1.5)
    Z = model.fit_transform(X)

    assert Z.shape == (4, 2)
    assert np.all(model.eigenvalues_ > 0.0)
    assert np.allclose(np.mean(Z, axis=0), 0.0, atol=1e-7)


def test_kernel_pca_transforms_new_samples():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    model = KernelPCA(n_components=2, kernel="poly", degree=2, coef0=0.5).fit(X)

    out = model.transform([[1.5], [2.5]])

    assert out.shape == (2, 2)
