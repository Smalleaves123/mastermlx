import numpy as np

from mastermlx.linear_models import (
    HuberRegressor,
    Perceptron,
    SGDClassifier,
    SGDRegressor,
)


# ---------------------------------------------------------------------------
# Perceptron
# ---------------------------------------------------------------------------


def test_perceptron_linearly_separable():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 0, 0, 1])  # AND: only (1,1) is 1
    p = Perceptron(max_iter=100, random_state=0).fit(X, y)
    assert np.all(p.predict(X) == y)


def test_perceptron_decision():
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    p = Perceptron(max_iter=100, random_state=0).fit(X, y)
    assert p.decision_function([[0.5]]).ndim == 1


# ---------------------------------------------------------------------------
# SGDClassifier
# ---------------------------------------------------------------------------


def test_sgd_clf_hinge_binary():
    X = np.random.randn(200, 5)
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, 0)
    clf = SGDClassifier(loss="hinge", max_iter=50, tol=1e-4, random_state=0).fit(X, y)
    acc = np.mean(clf.predict(X) == y)
    assert acc > 0.85


def test_sgd_clf_log_loss():
    X = np.random.randn(200, 3)
    y = np.where(X[:, 0] - X[:, 1] > 0, 1, 0)
    clf = SGDClassifier(loss="log_loss", max_iter=50, tol=1e-4, random_state=0).fit(X, y)
    acc = np.mean(clf.predict(X) == y)
    assert acc > 0.80


def test_sgd_clf_multiclass():
    X = np.random.randn(300, 3)
    y = np.zeros(300, dtype=int)
    y[X[:, 0] > 0.5] = 1
    y[X[:, 0] < -0.5] = 2
    clf = SGDClassifier(loss="log_loss", max_iter=50, random_state=0).fit(X, y)
    scores = clf.decision_function(X)
    assert scores.shape == (300, 3)
    acc = np.mean(clf.predict(X) == y)
    assert acc > 0.80


def test_sgd_clf_l1_penalty():
    X = np.random.randn(200, 50)
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, 0)
    clf1 = SGDClassifier(penalty="l2", alpha=0.1, max_iter=50, random_state=0).fit(X, y)
    clf2 = SGDClassifier(penalty="l1", alpha=0.1, max_iter=50, random_state=0).fit(X, y)
    # L1 coefs should be smaller in L1 norm (promotes sparsity)
    assert np.sum(np.abs(clf2.coef_)) < np.sum(np.abs(clf1.coef_)) * 0.95


# ---------------------------------------------------------------------------
# SGDRegressor
# ---------------------------------------------------------------------------


def test_sgd_reg_squared():
    X = np.random.randn(200, 3)
    coef = np.array([2.0, -1.0, 0.5])
    y = X @ coef + 0.1 * np.random.randn(200)
    reg = SGDRegressor(loss="squared_error", max_iter=100, eta0=0.01, random_state=0).fit(X, y)
    pred = reg.predict(X)
    assert np.corrcoef(pred, y)[0, 1] > 0.95


def test_sgd_reg_huber():
    X = np.random.randn(200, 3)
    coef = np.array([2.0, -1.0, 0.5])
    y = X @ coef + 0.1 * np.random.randn(200)
    y[0] = 500.0  # outlier
    reg = SGDRegressor(loss="huber", delta=1.0, max_iter=100, eta0=0.01, random_state=0).fit(X, y)
    # With huber loss, outlier shouldn't ruin fit
    assert np.corrcoef(reg.predict(X[1:]), y[1:])[0, 1] > 0.90


# ---------------------------------------------------------------------------
# HuberRegressor (IRLS)
# ---------------------------------------------------------------------------


def test_huber_clean_data():
    X = np.random.randn(100, 3)
    coef = np.array([2.0, -1.0, 0.5])
    y = X @ coef + 0.05 * np.random.randn(100)
    hr = HuberRegressor().fit(X, y)
    pred = hr.predict(X)
    assert np.corrcoef(pred, y)[0, 1] > 0.95


def test_huber_outliers():
    X = np.random.randn(200, 3)
    coef = np.array([2.0, -1.0, 0.5])
    y = X @ coef + 0.05 * np.random.randn(200)
    y[:10] = 500.0  # 5% outliers
    hr = HuberRegressor(epsilon=1.35).fit(X, y)
    # Outliers should be flagged
    assert len(hr.outliers_) >= 5
    # Good points should still fit well
    good = hr.predict(X[10:])
    assert np.corrcoef(good, y[10:])[0, 1] > 0.90
