import numpy as np

from mastermlx import LogisticRegression, StandardScaler, LinearRegression
from mastermlx.tabular import TabularExperiment, compare_tabular_models
from mastermlx.tabular.workflow import TabularExperiment as WorkflowExperiment


def test_tabular_facade_exports_workflow_implementation():
    assert TabularExperiment is WorkflowExperiment


def test_tabular_experiment_runs_preprocessing_and_grid_search():
    X = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.8, 1.0],
            [1.0, 0.9],
            [0.1, 0.2],
            [0.9, 1.1],
        ]
    )
    y = np.array([0, 0, 1, 1, 0, 1])

    experiment = TabularExperiment(
        model=LogisticRegression(n_iter=500, random_state=0),
        preprocessing=StandardScaler(),
        search="grid",
        param_grid={"model__lr": [0.01, 0.1]},
        cv=3,
    )
    experiment.fit(X, y)

    pred = experiment.predict(X)
    assert pred.shape == y.shape
    assert experiment.best_estimator_ is not None
    assert experiment.summary()["task"] == "classification"
    assert experiment.best_score_ is not None
    assert 0.0 <= experiment.score(X, y) <= 1.0


def test_compare_tabular_models_returns_leaderboard():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ]
    )
    y = np.array([0.0, 1.0, 2.0, 3.0])

    result = compare_tabular_models(
        [
            ("linear_a", LinearRegression(fit_intercept=True)),
            ("linear_b", LinearRegression(fit_intercept=False)),
        ],
        X,
        y,
        task="regression",
    )

    assert result["leaderboard"]
    assert result["best_name"] in {"linear_a", "linear_b"}
    assert result["best_experiment"] is not None
    assert result["best_score"] == result["best_experiment"].best_score_


def test_tabular_cv_score_uses_fitted_pipeline_contract():
    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = 2.0 * X.ravel() + 1.0
    experiment = TabularExperiment(model=LinearRegression(), search=None, cv=3, task="regression")
    experiment.fit(X, y)
    scores = experiment.cv_score(X, y)

    assert scores.shape == (3,)
    assert np.all(scores > 0.99)
