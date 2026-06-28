import numpy as np

from mastermlx.linear_models import ElasticNetRegression, LassoRegression, RidgeRegression


def test_ridge_regression_fits_stable_line():
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y = 4.0 * X.ravel() + 1.0

    model = RidgeRegression(alpha=0.1)
    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert model.score(X, y) > 0.99
    assert np.isfinite(model.intercept_)


def test_lasso_regression_can_zero_irrelevant_feature():
    X = np.array([
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
    ])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    model = LassoRegression(alpha=0.05, max_iter=5000, tol=1e-6)
    model.fit(X, y)

    assert model.score(X, y) > 0.98
    assert abs(model.coef_[1]) < 1e-8


def test_elastic_net_regression_fits_simple_data():
    X = np.array([
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [3.0, 1.0],
        [4.0, 1.0],
    ])
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    model = ElasticNetRegression(alpha=0.05, l1_ratio=0.5, max_iter=5000, tol=1e-6)
    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert model.score(X, y) > 0.98
