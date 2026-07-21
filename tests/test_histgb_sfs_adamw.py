import numpy as np
import pytest

from mastermlx.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from mastermlx.ensemble.hist_gb import _fit_hist_cpp, _predict_hist_cpp
from mastermlx.neural_net import AdamW
from mastermlx.selection import SequentialFeatureSelector
from mastermlx.linear_models import LogisticRegression
from mastermlx import get_backend, set_backend


# --- HistGradientBoosting ---


def test_histgb_clf_binary():
    X = np.random.default_rng(0).normal(size=(300, 10))
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, 0)
    clf = HistGradientBoostingClassifier(n_estimators=20, max_depth=4, random_state=0).fit(X, y)
    assert clf.score(X, y) > 0.80
    proba = clf.predict_proba(X)
    assert proba.shape == (300, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_histgb_clf_single_sample():
    X = np.random.default_rng(1).normal(size=(100, 5))
    y = np.where(X[:, 0] > 0, 1, 0)
    clf = HistGradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=0).fit(X, y)
    p = clf.predict(X[:1])
    assert p.shape == (1,)


def test_histgb_reg():
    X = np.random.default_rng(2).normal(size=(200, 5))
    coef = np.array([2.0, -1.0, 0.5, 0.0, 0.0])
    y = X @ coef + 0.1 * np.random.randn(200)
    reg = HistGradientBoostingRegressor(n_estimators=30, max_depth=3, random_state=0).fit(X, y)
    assert reg.score(X, y) > 0.85


def test_histgb_reg_single():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(100, 3))
    y = X[:, 0] * 2 + 0.1 * rng.normal(size=100)
    reg = HistGradientBoostingRegressor(n_estimators=10, max_depth=3, random_state=0).fit(X, y)
    assert isinstance(float(reg.predict(X[:1])[0]), float)


def test_histgb_numpy_and_compiled_paths_match():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(160, 5))
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    old = get_backend()
    try:
        set_backend("numpy")
        numpy_model = HistGradientBoostingClassifier(
            n_estimators=8, max_depth=3, min_samples_leaf=5, random_state=0
        ).fit(X, y)
        set_backend("auto")
        compiled_model = HistGradientBoostingClassifier(
            n_estimators=8, max_depth=3, min_samples_leaf=5, random_state=0
        ).fit(X, y)
        assert np.allclose(
            numpy_model.predict_proba(X), compiled_model.predict_proba(X), atol=1e-10
        )
    finally:
        set_backend(old)


def test_histgb_feature_subsampling_is_reproducible_and_backend_parity_holds():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(180, 6))
    y = (X[:, 0] - X[:, 1] + 0.2 * X[:, 2] > 0).astype(int)
    old = get_backend()
    try:
        set_backend("numpy")
        numpy_model = HistGradientBoostingClassifier(
            n_estimators=6, max_depth=3, min_samples_leaf=5, max_features=3, random_state=7
        ).fit(X, y)
        repeat_model = HistGradientBoostingClassifier(
            n_estimators=6, max_depth=3, min_samples_leaf=5, max_features=3, random_state=7
        ).fit(X, y)
        assert all(tree.feature_indices_.size == 3 for tree in numpy_model.trees_)
        assert np.allclose(numpy_model.predict_proba(X), repeat_model.predict_proba(X))
        assert numpy_model.get_params()["max_features"] == 3

        set_backend("auto")
        compiled_model = HistGradientBoostingClassifier(
            n_estimators=6, max_depth=3, min_samples_leaf=5, max_features=3, random_state=7
        ).fit(X, y)
        assert np.allclose(
            numpy_model.predict_proba(X), compiled_model.predict_proba(X), atol=1e-10
        )
    finally:
        set_backend(old)


def test_compiled_hist_tree_preserves_valid_child_indices():
    if _fit_hist_cpp is None or _predict_hist_cpp is None:
        pytest.skip("compiled histogram tree extension is unavailable")
    rng = np.random.default_rng(0)
    X = rng.integers(0, 8, size=(120, 4), dtype=np.int32)
    g = rng.normal(size=120)
    h = np.ones(120, dtype=float)
    nodes = _fit_hist_cpp(X, g, h, 3, 4, 0.1)
    features, bins, left, right, values = nodes
    n_nodes = len(features)
    assert len({len(features), len(bins), len(left), len(right), len(values)}) == 1
    for feature, child_left, child_right in zip(features, left, right):
        if feature >= 0:
            assert 0 <= child_left < n_nodes
            assert 0 <= child_right < n_nodes
        else:
            assert child_left == -1
            assert child_right == -1
    prediction = _predict_hist_cpp(X, *nodes)
    assert prediction.shape == (X.shape[0],)


def test_compiled_hist_tree_rejects_invalid_child_indices():
    if _predict_hist_cpp is None:
        pytest.skip("compiled histogram tree extension is unavailable")
    X = np.zeros((2, 1), dtype=np.int32)
    features = np.array([0], dtype=np.int32)
    bins = np.array([0], dtype=np.int32)
    left = np.array([4], dtype=np.int32)
    right = np.array([-1], dtype=np.int32)
    values = np.array([0.0], dtype=float)
    with pytest.raises(ValueError, match="child index"):
        _predict_hist_cpp(X, features, bins, left, right, values)


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
    X = np.random.default_rng(4).normal(size=(200, 10))
    y = np.where(X[:, 0] + X[:, 1] - X[:, 2] > 0, 1, 0)
    sfs = SequentialFeatureSelector(
        LogisticRegression(lr=0.1, n_iter=50), n_features_to_select=3, direction="forward", cv=2
    )
    sfs.fit(X, y)
    assert sfs.support_.sum() == 3
    assert sfs.transform(X).shape == (200, 3)


def test_sfs_backward():
    X = np.random.default_rng(5).normal(size=(150, 8))
    y = np.where(X[:, 0] - X[:, 1] > 0, 1, 0)
    sfs = SequentialFeatureSelector(
        LogisticRegression(lr=0.1, n_iter=50), n_features_to_select=4, direction="backward", cv=2
    )
    sfs.fit(X, y)
    assert sfs.support_.sum() == 4
