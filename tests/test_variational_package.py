import numpy as np

import mastermlx
from mastermlx.clustering import BayesianGaussianMixture
from mastermlx.variational import (
    VariationalGaussianMixture,
    VariationalLinearRegression,
    VariationalLogisticRegression,
    VariationalPoissonRegression,
)


def test_mastermlx_variational_exports_are_available():
    assert hasattr(mastermlx, "VariationalGaussianMixture")
    assert hasattr(mastermlx, "VariationalLinearRegression")
    assert hasattr(mastermlx, "VariationalLogisticRegression")
    assert hasattr(mastermlx, "VariationalPoissonRegression")


def test_variational_linear_regression_exposes_unified_summary():
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y = 3.0 * X.ravel() + 2.0

    model = VariationalLinearRegression(max_iter=200, tol=1e-6).fit(X, y)
    summary = model.posterior_summary()
    trace = model.elbo_trace()

    assert trace.ndim == 1
    assert trace.shape[0] == model.n_iter_
    assert np.isclose(trace[-1], model.final_elbo())
    assert summary["model"] == "VariationalLinearRegression"
    assert "noise_shape" in summary
    assert "final_elbo" in summary


def test_variational_gaussian_mixture_exposes_unified_summary():
    X = np.array([
        [0.0, 0.0],
        [0.1, -0.1],
        [5.0, 5.0],
        [5.1, 4.9],
    ])

    model = VariationalGaussianMixture(n_components=2, max_iter=100, random_state=0).fit(X)
    summary = model.posterior_summary()

    assert summary["model"] == "VariationalGaussianMixture"
    assert summary["n_components"] == 2
    assert "min_weight" in summary


def test_bayesian_gmm_summary_tracks_active_components():
    X = np.array([
        [-0.1, 0.0],
        [0.0, 0.1],
        [5.0, 5.0],
        [5.1, 4.9],
    ])

    model = BayesianGaussianMixture(n_components=4, max_iter=100, random_state=0).fit(X)
    summary = model.posterior_summary()

    assert summary["model"] == "BayesianGaussianMixture"
    assert "n_active_components" in summary


def test_variational_logistic_regression_exposes_unified_summary():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    model = VariationalLogisticRegression(max_iter=200, tol=1e-6).fit(X, y)
    summary = model.posterior_summary()
    trace = model.elbo_trace()

    assert model.score(X, y) >= 0.75
    assert trace.shape[0] == model.n_iter_
    assert np.isclose(trace[-1], model.final_elbo())
    assert summary["model"] == "VariationalLogisticRegression"
    assert "alpha" in summary


def test_variational_poisson_regression_exposes_unified_summary():
    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = np.array([1, 1, 2, 3, 5, 8])

    model = VariationalPoissonRegression(max_iter=250, lr=0.02, tol=1e-6).fit(X, y)
    summary = model.posterior_summary()
    trace = model.elbo_trace()

    assert trace.shape[0] == model.n_iter_
    assert np.isclose(trace[-1], model.final_elbo())
    assert summary["model"] == "VariationalPoissonRegression"
    assert "avg_posterior_var" in summary
