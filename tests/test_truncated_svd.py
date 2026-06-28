import numpy as np

from mastermlx.decomposition import TruncatedSVD


def test_truncated_svd_reduces_dimension():
    X = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 3.0],
            [2.0, 1.0, 4.0],
        ]
    )

    model = TruncatedSVD(n_components=2)
    Z = model.fit_transform(X)

    assert Z.shape == (4, 2)
    assert model.components_.shape == (2, 3)
    assert model.singular_values_.shape == (2,)


def test_truncated_svd_inverse_transform_shape():
    X = np.array(
        [
            [1.0, 2.0, 0.0],
            [2.0, 1.0, 1.0],
            [3.0, 0.0, 2.0],
        ]
    )

    model = TruncatedSVD(n_components=2).fit(X)
    Z = model.transform(X)
    X2 = model.inverse_transform(Z)

    assert X2.shape == X.shape


def test_truncated_svd_reports_variance_ratio():
    X = np.array(
        [
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [4.0, 2.0],
        ]
    )

    model = TruncatedSVD(n_components=1).fit(X)

    assert model.explained_variance_.shape == (1,)
    assert model.explained_variance_ratio_.shape == (1,)
    assert model.explained_variance_ratio_[0] > 0.0
