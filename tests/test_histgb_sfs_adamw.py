import numpy as np

from mastermlx.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from mastermlx.neural_net import AdamW
from mastermlx.selection import SequentialFeatureSelector
from mastermlx.linear_models import LogisticRegression


# --- HistGradientBoosting ---

def test_histgb_clf_binary():
    X = np.random.randn(300, 10)
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, 0)
    clf = HistGradientBoostingClassifier(n_estimators=20, max_depth=4, random_state=0).fit(X, y)
    assert clf.score(X, y) > 0.80
    proba = clf.predict_proba(X)
    assert proba.shape == (300, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_histgb_clf_single_sample():
    X = np.random.randn(100, 5)
    y = np.where(X[:, 0] > 0, 1, 0)
    clf = HistGradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=0).fit(X, y)
    p = clf.predict(X[:1])
    assert p.shape == (1,)


def test_histgb_reg():
    X = np.random.randn(200, 5)
    coef = np.array([2.0, -1.0, 0.5, 0.0, 0.0])
    y = X @ coef + 0.1 * np.random.randn(200)
    reg = HistGradientBoostingRegressor(n_estimators=30, max_depth=3, random_state=0).fit(X, y)
    assert reg.score(X, y) > 0.85


def test_histgb_reg_single():
    X = np.random.randn(100, 3)
    y = X[:, 0] * 2 + 0.1 * np.random.randn(100)
    reg = HistGradientBoostingRegressor(n_estimators=10, max_depth=3, random_state=0).fit(X, y)
    assert isinstance(float(reg.predict(X[:1])[0]), float)


# --- AdamW ---

def test_adamw_updates():
    import numpy as np
    opt = AdamW(lr=0.01, weight_decay=0.01)
    w = np.ones(5, dtype=float)
    grad = np.array([0.1, 0.2, -0.1, 0.0, 0.05])
    w2 = opt.update(w, grad, "w1")
    # Should decay and move
    assert not np.allclose(w, w2)
    # Should be smaller due to decay
    assert np.mean(np.abs(w2)) < 1.01


# --- SequentialFeatureSelector ---

def test_sfs_forward():
    X = np.random.randn(200, 10)
    y = np.where(X[:, 0] + X[:, 1] - X[:, 2] > 0, 1, 0)
    sfs = SequentialFeatureSelector(LogisticRegression(lr=0.1, n_iter=50), n_features_to_select=3,
                                    direction="forward", cv=2)
    sfs.fit(X, y)
    assert sfs.support_.sum() == 3
    assert sfs.transform(X).shape == (200, 3)


def test_sfs_backward():
    X = np.random.randn(150, 8)
    y = np.where(X[:, 0] - X[:, 1] > 0, 1, 0)
    sfs = SequentialFeatureSelector(LogisticRegression(lr=0.1, n_iter=50), n_features_to_select=4,
                                    direction="backward", cv=2)
    sfs.fit(X, y)
    assert sfs.support_.sum() == 4
