import numpy as np

from mastermlx import LogisticRegression, StandardScaler, LinearRegression
from mastermlx.tabular import TabularExperiment, compare_tabular_models


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
