import numpy as np
import pytest

from mastermlx.data import partial_dependence, permutation_importance
from mastermlx.linear_models import LinearRegression, LogisticRegression
from mastermlx.tabular import TabularExperiment


def test_permutation_importance_ranks_predictive_feature_and_is_reproducible():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 2))
    y = 4.0 * X[:, 0] + rng.normal(scale=0.05, size=X.shape[0])
    model = LinearRegression().fit(X, y)

    first = permutation_importance(
        model,
        X,
        y,
        scoring="r2",
        n_repeats=6,
        random_state=11,
        feature_names=["signal", "noise"],
    )
    second = permutation_importance(
        model,
        X,
        y,
        scoring="r2",
        n_repeats=6,
        random_state=11,
        feature_names=["signal", "noise"],
    )

    assert first["items"][0]["feature"] == "signal"
    assert first["importances_mean"][0] > 1.0
    assert abs(first["importances_mean"][1]) < 0.01
    assert np.array_equal(first["importances"], second["importances"])


def test_tabular_report_can_include_permutation_importance():
    X = np.array([[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    experiment = TabularExperiment(
        LogisticRegression(lr=0.5, n_iter=800, random_state=0),
        search=None,
        random_state=5,
    ).fit(X, y)

    report = experiment.report(
        X,
        y,
        include_permutation_importance=True,
        permutation_repeats=4,
    )

    assert report["permutation_importance"]["kind"] == "permutation"
    assert report["permutation_importance"]["n_repeats"] == 4
    assert report["permutation_importance"]["items"][0]["feature"] == "x0"


def test_permutation_importance_validates_inputs():
    model = LinearRegression().fit([[0.0], [1.0]], [0.0, 1.0])
    with pytest.raises(ValueError, match="n_repeats"):
        permutation_importance(model, [[0.0], [1.0]], [0.0, 1.0], n_repeats=0)
    with pytest.raises(ValueError, match="feature_names"):
        permutation_importance(
            model,
            [[0.0], [1.0]],
            [0.0, 1.0],
            feature_names=["duplicate", "duplicate"],
        )


def test_partial_dependence_returns_average_and_centered_ice_curves():
    X = np.column_stack([np.linspace(-2.0, 2.0, 60), np.ones(60)])
    y = 3.0 * X[:, 0] + 2.0
    model = LinearRegression().fit(X, y)

    result = partial_dependence(
        model,
        X,
        0,
        grid_values=[-1.0, 0.0, 1.0],
        kind="both",
        center=True,
    )

    assert result["feature"] == "x0"
    assert result["average"].shape == (3,)
    assert result["individual"].shape == (3, 60)
    assert np.allclose(result["average"], [0.0, 3.0, 6.0])
    assert np.allclose(result["individual"][0], 0.0)


def test_tabular_report_retains_partial_dependence():
    X = np.linspace(-1.0, 1.0, 30).reshape(-1, 1)
    y = 2.0 * X.ravel()
    experiment = TabularExperiment(
        LinearRegression(), search=None, task="regression"
    ).fit(X, y)

    result = experiment.partial_dependence(0, grid_resolution=5)
    report = experiment.report(X, y)

    assert result["average"].shape == (5,)
    assert report["partial_dependence"]["x0"] is result


def test_partial_dependence_validates_multiclass_target_and_kind():
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.array([0, 1, 2, 0, 1, 2])
    model = LogisticRegression(n_iter=50, random_state=0).fit(X, y)

    with pytest.raises(ValueError, match="target is required"):
        partial_dependence(model, X, 0)
    with pytest.raises(ValueError, match="kind"):
        partial_dependence(model, X, 0, kind="unknown", target=1)
