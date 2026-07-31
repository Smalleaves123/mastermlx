import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.cluster import KMeans as SklearnKMeans  # noqa: E402
from sklearn.decomposition import PCA as SklearnPCA  # noqa: E402
from sklearn.linear_model import LinearRegression as SklearnLinearRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler  # noqa: E402

from mastermlx.clustering import KMeans  # noqa: E402
from mastermlx.decomposition import PCA  # noqa: E402
from mastermlx.linear_models import LinearRegression  # noqa: E402
from mastermlx.preprocessing import StandardScaler  # noqa: E402


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_linear_regression_matches_sklearn(fit_intercept):
    X = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [3.0, 7.0]])
    y = np.array([1.0, 3.0, 6.0, 10.0])

    ours = LinearRegression(fit_intercept=fit_intercept).fit(X, y)
    reference = SklearnLinearRegression(fit_intercept=fit_intercept).fit(X, y)

    np.testing.assert_allclose(ours.coef_, reference.coef_, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ours.intercept_, reference.intercept_, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ours.predict(X), reference.predict(X), rtol=1e-10, atol=1e-10)


def test_standard_scaler_matches_sklearn():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [5.0, 40.0], [8.0, 80.0]])

    ours = StandardScaler().fit(X)
    reference = SklearnStandardScaler().fit(X)

    np.testing.assert_allclose(ours.mean_, reference.mean_)
    np.testing.assert_allclose(ours.scale_, reference.scale_)
    np.testing.assert_allclose(ours.transform(X), reference.transform(X))


def test_pca_matches_sklearn_invariant_quantities():
    X = np.array([
        [1.0, 2.0, 0.0],
        [2.0, 4.0, 1.0],
        [3.0, 6.0, 2.0],
        [4.0, 8.0, 3.0],
        [5.0, 10.0, 4.0],
    ])

    ours = PCA(n_components=2).fit(X)
    reference = SklearnPCA(n_components=2).fit(X)

    np.testing.assert_allclose(ours.mean_, reference.mean_)
    np.testing.assert_allclose(ours.explained_variance_, reference.explained_variance_)
    np.testing.assert_allclose(ours.explained_variance_ratio_, reference.explained_variance_ratio_)
    np.testing.assert_allclose(
        np.abs(ours.transform(X)),
        np.abs(reference.transform(X)),
        rtol=1e-8,
        atol=1e-8,
    )


def test_kmeans_matches_sklearn_inertia_and_partition():
    X = np.array([
        [-3.0, -2.0], [-2.5, -2.0], [-2.0, -3.0],
        [2.0, 2.0], [2.5, 2.0], [3.0, 3.0],
    ])

    ours = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
    reference = SklearnKMeans(n_clusters=2, n_init=10, random_state=0).fit(X)

    assert np.isclose(ours.inertia_, reference.inertia_)
    assert np.array_equal(ours.predict(X), ours.labels_)
    assert np.array_equal(
        np.sort(np.bincount(ours.labels_)),
        np.sort(np.bincount(reference.labels_)),
    )
