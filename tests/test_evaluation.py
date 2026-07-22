import numpy as np

from mastermlx import LinearRegression, LogisticRegression
from mastermlx.data import (
    EvaluationReport,
    GroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    compare_estimators,
)
from mastermlx.tabular import TabularExperiment


def test_evaluation_report_builds_oof_bootstrap_and_learning_curve():
    X = np.arange(18, dtype=float).reshape(-1, 1)
    y = (X[:, 0] > 8).astype(int)
    report = EvaluationReport(
        LogisticRegression(lr=0.4, n_iter=1000, random_state=0),
        task="classification",
        random_state=0,
    ).run(
        X,
        y,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
        n_bootstrap=80,
        train_sizes=[0.5, 1.0],
    )

    assert report["cv"]["name"] == "StratifiedKFold"
    assert report["oof"]["prediction"]
    assert len(report["oof"]["prediction"]) == X.shape[0]
    assert "accuracy" in report["bootstrap"]
    assert report["bootstrap"]["accuracy"]["n_resamples"] == 80
    assert report["learning_curve"]["train_sizes"] == [6, 12]


def test_evaluation_report_supports_group_and_time_series_cv():
    X = np.arange(24, dtype=float).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 1.0
    groups = np.repeat(np.arange(6), 4)

    group_report = EvaluationReport(LinearRegression(), task="regression").run(
        X,
        y,
        cv=GroupKFold(n_splits=3),
        groups=groups,
        n_bootstrap=20,
        include_learning_curve=False,
    )
    time_report = EvaluationReport(LinearRegression(), task="regression").run(
        X,
        y,
        cv=TimeSeriesSplit(n_splits=3, test_size=4),
        n_bootstrap=20,
        include_learning_curve=False,
    )

    assert group_report["cv"]["name"] == "GroupKFold"
    assert time_report["cv"]["name"] == "TimeSeriesSplit"
    assert len(time_report["cv"]["test_scores"]["r2"]) == 3
    assert time_report["oof"]["metrics"]["r2"] > 0.99


def test_compare_estimators_uses_paired_fold_significance():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = 3.0 * X[:, 0] + 2.0
    result = compare_estimators(
        [("with_intercept", LinearRegression(fit_intercept=True)), ("without_intercept", LinearRegression(fit_intercept=False))],
        X,
        y,
        task="regression",
        cv=4,
        random_state=0,
        n_permutations=200,
    )

    assert result["primary_metric"] == "r2"
    assert len(result["leaderboard"]) == 2
    assert len(result["pairwise"]) == 1
    assert "p_value" in result["pairwise"][0]


def test_tabular_experiment_exposes_evaluation_report():
    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 1.0
    experiment = TabularExperiment(LinearRegression(), search=None, task="regression").fit(X, y)
    report = experiment.evaluation_report(n_bootstrap=20, include_learning_curve=False)

    assert report["oof"]["metrics"]["r2"] > 0.99
    assert report["cv"]["n_splits"] == 5
