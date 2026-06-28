import numpy as np

from mastermlx.manifold import TSNE
from mastermlx.svm import NuSVC
from mastermlx.probabilistic import KernelDensity
from mastermlx.linear_models import QuantileRegressor, RANSACRegressor


# --- t-SNE ---
def test_tsne_fit_shape():
    X = np.random.randn(50, 10)
    tsne = TSNE(n_components=2, perplexity=10, n_iter=100, random_state=0)
    Y = tsne.fit_transform(X)
    assert Y.shape == (50, 2)
    assert tsne.kl_divergence_ > 0


def test_tsne_reproducible():
    X = np.random.randn(30, 5)
    Y1 = TSNE(n_iter=100, random_state=42).fit_transform(X)
    Y2 = TSNE(n_iter=100, random_state=42).fit_transform(X)
    assert np.allclose(Y1, Y2)


# --- NuSVC ---
def test_nusvc_binary():
    X = np.vstack([np.random.randn(30, 2) + [0, 0], np.random.randn(30, 2) + [3, 3]])
    y = np.array([0]*30 + [1]*30)
    clf = NuSVC(nu=0.3, kernel="linear", max_iter=500, random_state=0).fit(X, y)
    assert clf.score(X, y) > 0.85


def test_nusvc_multiclass():
    X = np.vstack([np.random.randn(20, 2)+[0,0], np.random.randn(20, 2)+[4,0], np.random.randn(20, 2)+[2,4]])
    y = np.array([0]*20 + [1]*20 + [2]*20)
    clf = NuSVC(nu=0.5, kernel="linear", max_iter=500, random_state=0).fit(X, y)
    assert clf.score(X, y) > 0.80


# --- KernelDensity ---
def test_kde_score_samples():
    X = np.random.randn(100, 2)
    kde = KernelDensity(bandwidth=0.5).fit(X)
    scores = kde.score_samples(X)
    assert scores.shape == (100,)
    assert np.all(np.isfinite(scores))


def test_kde_sample():
    X = np.random.randn(100, 2)
    kde = KernelDensity(bandwidth=0.5).fit(X)
    samples = kde.sample(30, random_state=0)
    assert samples.shape == (30, 2)


# --- QuantileRegressor ---
def test_quantile_median():
    X = np.random.randn(100, 3)
    coef = np.array([2.0, -1.0, 0.5])
    y = X @ coef + np.random.randn(100) * 0.5
    qr = QuantileRegressor(quantile=0.5).fit(X, y)
    pred = qr.predict(X)
    assert np.corrcoef(pred, y)[0, 1] > 0.9


def test_quantile_upper():
    X = np.random.randn(100, 2)
    y = X[:, 0] * 2 + np.random.exponential(1, 100)
    qr = QuantileRegressor(quantile=0.9).fit(X, y)
    # 90th percentile predictions should be above ~80% of data
    pred = qr.predict(X)
    assert np.mean(pred > y) > 0.7


# --- RANSAC ---
def test_ransac_clean():
    X = np.random.randn(100, 2)
    y = X[:, 0] * 3.0 - X[:, 1] * 2.0 + 0.5
    rr = RANSACRegressor(random_state=0).fit(X, y)
    assert np.corrcoef(rr.predict(X), y)[0, 1] > 0.95


def test_ransac_outliers():
    X = np.random.randn(200, 2)
    y = X[:, 0] * 3.0 + 1.0
    y[:40] += np.random.randn(40) * 50  # 20% gross outliers
    rr = RANSACRegressor(residual_threshold=5.0, random_state=0).fit(X, y)
    assert rr.inlier_mask_ is not None
    assert np.sum(rr.inlier_mask_) < 200  # outliers excluded
    # Good inliers should fit well
    assert np.corrcoef(rr.predict(X[rr.inlier_mask_]), y[rr.inlier_mask_])[0, 1] > 0.9
