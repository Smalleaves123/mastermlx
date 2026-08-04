import numpy as np

from mastermlx import get_backend, set_backend
from mastermlx.clustering import MiniBatchKMeans
from mastermlx.preprocessing import KNNImputer
from mastermlx.neighbors import NearestCentroid
from mastermlx.decomposition import CCA


# --- MiniBatchKMeans ---
def test_minibatch_kmeans_shape():
    X = np.random.randn(500, 5)
    mb = MiniBatchKMeans(n_clusters=3, batch_size=50, max_iter=20, random_state=0).fit(X)
    assert mb.cluster_centers_.shape == (3, 5)
    assert mb.predict(X).shape == (500,)
    assert mb.inertia_ >= 0


def test_minibatch_kmeans_single_init():
    X = np.random.randn(200, 3)
    mb = MiniBatchKMeans(n_clusters=2, n_init=1, random_state=0).fit(X)
    assert mb.labels_.shape == (200,)
    assert len(np.unique(mb.labels_)) == 2


def test_minibatch_kmeans_cpp_and_numpy_match():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 4)).astype(np.float32)
    old = get_backend()
    try:
        set_backend("auto")
        accelerated = MiniBatchKMeans(
            n_clusters=3, batch_size=20, max_iter=8, n_init=2, random_state=7
        ).fit(X)
        set_backend("numpy")
        fallback = MiniBatchKMeans(
            n_clusters=3, batch_size=20, max_iter=8, n_init=2, random_state=7
        ).fit(X)
    finally:
        set_backend(old)

    assert np.allclose(accelerated.cluster_centers_, fallback.cluster_centers_)
    assert np.array_equal(accelerated.labels_, fallback.labels_)
    assert np.allclose(accelerated.inertia_, fallback.inertia_)


# --- KNNImputer ---
def test_knni_basic():
    X = np.random.randn(50, 4)
    X[0, 0] = np.nan
    X[5, 2] = np.nan
    imp = KNNImputer(n_neighbors=3).fit(X)
    Xt = imp.transform(X)
    assert not np.any(np.isnan(Xt))
    assert Xt.shape == X.shape


def test_knni_all_complete():
    X = np.random.randn(30, 3)
    imp = KNNImputer().fit(X)
    Xt = imp.transform(X)
    assert np.allclose(X, Xt)


def test_knni_handles_missing_values_in_every_feature():
    X = np.arange(20, dtype=float).reshape(5, 4)
    X[0, 0] = np.nan
    X[1, 1] = np.nan
    X[2, 2] = np.nan
    X[3, 3] = np.nan

    Xt = KNNImputer(n_neighbors=2, weights="uniform").fit_transform(X)

    assert np.isfinite(Xt).all()
    assert Xt.shape == X.shape


def test_knni_transform_supports_query_size_different_from_training_size():
    X_fit = np.arange(20, dtype=float).reshape(5, 4)
    query = np.array([[np.nan, 1.0, 2.0, 3.0], [4.0, 5.0, np.nan, 7.0]])
    imputer = KNNImputer(n_neighbors=2).fit(X_fit)

    transformed = imputer.transform(query)

    assert transformed.shape == query.shape
    assert np.isfinite(transformed).all()


# --- NearestCentroid ---
def test_centroid_basic():
    X = np.vstack([np.random.randn(30, 3) + [0, 0, 0],
                   np.random.randn(30, 3) + [5, 0, 0]])
    y = np.array([0]*30 + [1]*30)
    nc = NearestCentroid().fit(X, y)
    assert nc.score(X, y) > 0.90


def test_centroid_single_sample():
    X = np.random.randn(20, 2)
    y = np.where(X[:, 0] > 0, 1, 0)
    nc = NearestCentroid().fit(X, y)
    assert nc.predict(X[:1]).shape == (1,)


# --- CCA ---
def test_cca_shape():
    X = np.random.randn(100, 8)
    Y = np.random.randn(100, 5)
    Y[:, 0] = X[:, 0] * 2 + X[:, 1] + 0.1 * np.random.randn(100)
    cca = CCA(n_components=2).fit(X, Y)
    Zx = cca.transform(X)
    assert Zx.shape == (100, 2)
    assert cca.corrs_.shape == (2,)
    assert np.all(cca.corrs_ >= 0) and np.all(cca.corrs_ <= 1)


def test_cca_max_corr():
    X = np.random.randn(200, 4)
    Y = np.column_stack([X[:, 0]*3 + np.random.randn(200)*0.01, np.random.randn(200)])
    cca = CCA(n_components=1).fit(X, Y)
    assert cca.corrs_[0] > 0.95
