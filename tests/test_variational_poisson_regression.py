import numpy as np

from mastermlx.variational import VariationalPoissonRegression


def test_variational_poisson_regression_fits_increasing_count_trend():
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y = np.array([1, 1, 2, 3, 4, 6, 8, 11])

    model = VariationalPoissonRegression(alpha=1.0, max_iter=400, lr=0.02, tol=1e-6).fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert np.all(pred > 0.0)
    assert model.score(X, y) > 0.7


def test_variational_poisson_regression_returns_predictive_std():
    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = np.array([1, 1, 2, 3, 5, 8])

    model = VariationalPoissonRegression(max_iter=400, lr=0.02).fit(X, y)
    mean, std = model.predict([[2.5]], return_std=True)

    assert isinstance(mean, float)
    assert isinstance(std, float)
    assert mean > 0.0
    assert std > 0.0


def test_variational_poisson_regression_samples_posterior_weights():
    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = np.array([1, 1, 2, 3, 5, 8])

    model = VariationalPoissonRegression(max_iter=400, lr=0.02).fit(X, y)
    weights = model.sample_posterior_weights(n_samples=4, random_state=0)

    assert weights.shape == (4, 2)
