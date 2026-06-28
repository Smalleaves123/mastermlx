import numpy as np

from mastermlx.probabilistic import GaussianProcessRegressor


def test_gaussian_process_regressor_fits_smooth_curve():
    X = np.linspace(0.0, 2.0 * np.pi, 30).reshape(-1, 1)
    y = np.sin(X.ravel())

    model = GaussianProcessRegressor(length_scale=1.0, alpha=1e-5).fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert model.score(X, y) > 0.95


def test_gaussian_process_regressor_returns_std():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 0.0, -1.0])

    model = GaussianProcessRegressor(length_scale=0.8, alpha=1e-5).fit(X, y)
    mean, std = model.predict([[1.5]], return_std=True)

    assert isinstance(mean, float)
    assert isinstance(std, float)
    assert std >= 0.0


def test_gaussian_process_regressor_shapes_are_stored():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 0.0])

    model = GaussianProcessRegressor().fit(X, y)

    assert model.X_train_.shape == (3, 1)
    assert model.alpha_vec_.shape == (3,)


def test_gaussian_process_regressor_exposes_summary_and_posterior_sampling():
    X = np.linspace(0.0, 1.0, 5).reshape(-1, 1)
    y = np.sin(X.ravel())

    model = GaussianProcessRegressor(length_scale=0.7, alpha=1e-5).fit(X, y)
    summary = model.posterior_summary()
    samples = model.sample_posterior_functions([[0.25], [0.75]], n_samples=4, random_state=0)

    assert summary["model"] == "GaussianProcessRegressor"
    assert summary["n_train"] == 5
    assert samples.shape == (4, 2)
