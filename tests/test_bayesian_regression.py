import numpy as np

from mastermlx.probabilistic import BayesianLinearRegression, VariationalLinearRegression


def test_bayesian_linear_regression_fits_simple_line():
    X = np.arange(10, dtype=float).reshape(-1, 1)
    y = 2.0 * X.ravel() + 1.0

    model = BayesianLinearRegression(alpha=1.0, beta=10.0).fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert model.score(X, y) > 0.99


def test_bayesian_linear_regression_returns_predictive_std():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])

    model = BayesianLinearRegression(alpha=1.0, beta=5.0).fit(X, y)
    mean, std = model.predict([[1.5]], return_std=True)

    assert isinstance(mean, float)
    assert isinstance(std, float)
    assert std > 0.0


def test_bayesian_linear_regression_posterior_shapes():
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    y = np.array([1.0, 2.0, 4.0, 6.0])

    model = BayesianLinearRegression().fit(X, y)

    assert model.posterior_mean_.shape == (3,)
    assert model.posterior_cov_.shape == (3, 3)


def test_bayesian_linear_regression_exposes_summary_and_sampling():
    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = 2.0 * X.ravel() + 1.0

    model = BayesianLinearRegression(alpha=1.0, beta=10.0).fit(X, y)
    summary = model.posterior_summary()
    weights = model.sample_posterior_weights(n_samples=4, random_state=0)
    preds = model.sample_posterior_predictive([[1.5]], n_samples=3, random_state=0)

    assert summary["model"] == "BayesianLinearRegression"
    assert summary["posterior_dim"] == 2
    assert weights.shape == (4, 2)
    assert preds.shape == (3,)


def test_variational_linear_regression_fits_simple_line():
    X = np.arange(10, dtype=float).reshape(-1, 1)
    y = 2.0 * X.ravel() + 1.0

    model = VariationalLinearRegression(max_iter=300, tol=1e-6).fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert model.score(X, y) > 0.99


def test_variational_linear_regression_returns_predictive_std():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])

    model = VariationalLinearRegression(max_iter=300).fit(X, y)
    mean, std = model.predict([[1.5]], return_std=True)

    assert isinstance(mean, float)
    assert isinstance(std, float)
    assert std > 0.0


def test_variational_linear_regression_tracks_variational_state():
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    y = np.array([1.0, 2.0, 4.0, 6.0])

    model = VariationalLinearRegression(max_iter=300).fit(X, y)

    assert model.posterior_mean_.shape == (3,)
    assert model.posterior_cov_.shape == (3, 3)
    assert model.noise_shape_ > 0.0
    assert model.noise_rate_ > 0.0
    assert model.weight_shape_ > 0.0
    assert model.weight_rate_ > 0.0
    assert len(model.lower_bound_) >= 1


def test_variational_linear_regression_samples_posterior_weights_and_predictive():
    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = 2.0 * X.ravel() + 1.0

    model = VariationalLinearRegression(max_iter=300).fit(X, y)
    weights = model.sample_posterior_weights(n_samples=5, random_state=0)
    preds = model.sample_posterior_predictive([[1.5]], n_samples=4, random_state=0)

    assert weights.shape == (5, 2)
    assert preds.shape == (4,)
