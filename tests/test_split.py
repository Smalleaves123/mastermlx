import numpy as np

from mastermlx.data import (
    GroupKFold,
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_predict,
    cross_val_score,
    cross_validate,
    learning_curve,
    train_test_split,
    validation_curve,
)
from mastermlx.linear_models import LinearRegression, LogisticRegression


def test_train_test_split_shapes():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    assert X_train.shape == (7, 2)
    assert X_test.shape == (3, 2)
    assert y_train.shape == (7,)
    assert y_test.shape == (3,)
    assert np.array_equal(X_test, X[:3])


def test_kfold_covers_all_samples_once():
    X = np.arange(24).reshape(12, 2)
    cv = KFold(n_splits=3, shuffle=False)

    seen = []
    for train_idx, test_idx in cv.split(X):
        assert train_idx.shape[0] == 8
        assert test_idx.shape[0] == 4
        seen.extend(test_idx.tolist())

    assert sorted(seen) == list(range(12))


def test_stratified_kfold_preserves_class_balance():
    X = np.arange(24).reshape(12, 2)
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    cv = StratifiedKFold(n_splits=3, shuffle=False)

    for _, test_idx in cv.split(X, y):
        y_test = y[test_idx]
        assert np.sum(y_test == 0) == 2
        assert np.sum(y_test == 1) == 2


def test_cross_val_score_classification_uses_estimator_score():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression(lr=0.5, n_iter=3000, random_state=0)
    cv = KFold(n_splits=3, shuffle=True, random_state=0)

    scores = cross_val_score(model, X, y, cv=cv)

    assert scores.shape == (3,)
    assert np.all(scores >= 0.5)


def test_cross_val_score_regression_supports_named_metric():
    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = 3.0 * X.ravel() + 2.0
    model = LinearRegression()
    cv = KFold(n_splits=3, shuffle=False)

    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

    assert scores.shape == (3,)
    assert np.all(scores > 0.99)


def test_cross_val_score_accepts_callable_metric():
    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = 2.0 * X.ravel()
    model = LinearRegression()

    scores = cross_val_score(model, X, y, cv=KFold(n_splits=4), scoring=lambda yt, yp: -np.mean(np.abs(yt - yp)))

    assert scores.shape == (4,)


def test_cross_val_predict_returns_oof_predictions():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression(lr=0.5, n_iter=3000, random_state=0)

    pred = cross_val_predict(model, X, y, cv=KFold(n_splits=3, shuffle=True, random_state=0))

    assert pred.shape == y.shape
    assert set(np.unique(pred).tolist()) <= {0, 1}


def test_cross_val_predict_supports_predict_proba():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression(lr=0.5, n_iter=3000, random_state=0)

    proba = cross_val_predict(model, X, y, cv=KFold(n_splits=3, shuffle=True, random_state=0), method="predict_proba")

    assert proba.shape == (6, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-8)


def test_cross_validate_returns_scores_and_times():
    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = 3.0 * X.ravel() + 1.0
    model = LinearRegression()

    out = cross_validate(model, X, y, cv=KFold(n_splits=3), scoring="r2")

    assert set(out) == {"test_score", "fit_time", "score_time"}
    assert out["test_score"].shape == (3,)
    assert out["fit_time"].shape == (3,)
    assert out["score_time"].shape == (3,)


def test_cross_validate_can_return_train_scores():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression(lr=0.5, n_iter=3000, random_state=0)

    out = cross_validate(model, X, y, cv=KFold(n_splits=3, shuffle=True, random_state=0), return_train_score=True)

    assert set(out) == {"test_score", "train_score", "fit_time", "score_time"}
    assert out["train_score"].shape == (3,)


def test_cross_validate_supports_multiple_metrics():
    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = 3.0 * X.ravel() + 1.0
    model = LinearRegression()

    out = cross_validate(model, X, y, cv=KFold(n_splits=3), scoring=["r2", "neg_mean_squared_error"])

    assert set(out) == {"test_r2", "test_neg_mean_squared_error", "fit_time", "score_time"}
    assert out["test_r2"].shape == (3,)
    assert out["test_neg_mean_squared_error"].shape == (3,)


def test_cross_validate_supports_probability_metrics_and_estimators():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    out = cross_validate(
        LogisticRegression(lr=0.5, n_iter=2000, random_state=0),
        X,
        y,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
        scoring={"acc": "accuracy", "auc": "roc_auc", "loss": "neg_log_loss"},
        return_estimator=True,
    )

    assert out["test_acc"].shape == (3,)
    assert out["test_auc"].shape == (3,)
    assert out["test_loss"].shape == (3,)
    assert len(out["estimator"]) == 3
    assert np.all(np.isfinite(out["test_auc"]))


def test_time_series_split_keeps_time_order():
    X = np.arange(12).reshape(12, 1)
    cv = TimeSeriesSplit(n_splits=3, test_size=2)

    folds = list(cv.split(X))

    assert len(folds) == 3
    assert np.array_equal(folds[0][0], np.array([0, 1, 2, 3, 4, 5]))
    assert np.array_equal(folds[0][1], np.array([6, 7]))
    assert np.array_equal(folds[2][0], np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    assert np.array_equal(folds[2][1], np.array([10, 11]))


def test_time_series_split_gap_and_rolling_window_prevent_boundary_leakage():
    X = np.arange(12).reshape(12, 1)
    cv = TimeSeriesSplit(n_splits=3, test_size=2, gap=2, max_train_size=4)

    folds = list(cv.split(X))

    assert np.array_equal(folds[0][0], np.array([0, 1, 2, 3]))
    assert np.array_equal(folds[0][1], np.array([6, 7]))
    assert np.array_equal(folds[-1][0], np.array([4, 5, 6, 7]))
    assert np.array_equal(folds[-1][1], np.array([10, 11]))
    for train, test in folds:
        assert train[-1] + cv.gap < test[0]


def test_group_kfold_keeps_groups_separate():
    X = np.arange(16).reshape(8, 2)
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    cv = GroupKFold(n_splits=2)

    for train_idx, test_idx in cv.split(X, groups=groups):
        train_groups = set(groups[train_idx].tolist())
        test_groups = set(groups[test_idx].tolist())
        assert train_groups.isdisjoint(test_groups)


def test_shuffle_split_draws_requested_sizes():
    X = np.arange(20).reshape(10, 2)
    cv = ShuffleSplit(n_splits=3, test_size=0.2, train_size=0.5, random_state=0)

    splits = list(cv.split(X))

    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert train_idx.shape == (5,)
        assert test_idx.shape == (2,)
        assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0


def test_cross_validate_accepts_groups():
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    model = LogisticRegression(lr=0.5, n_iter=2000, random_state=0)

    out = cross_validate(model, X, y, cv=GroupKFold(n_splits=2), groups=groups)

    assert set(out) == {"test_score", "fit_time", "score_time"}
    assert out["test_score"].shape == (2,)


def test_learning_curve_returns_expected_shapes():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = 2.0 * X.ravel() + 1.0
    model = LinearRegression()

    sizes, train_scores, test_scores = learning_curve(
        model,
        X,
        y,
        train_sizes=[0.25, 0.5, 1.0],
        cv=KFold(n_splits=4, shuffle=False),
        scoring="r2",
    )

    assert np.array_equal(sizes, np.array([4, 8, 15]))
    assert train_scores.shape == (3, 4)
    assert test_scores.shape == (3, 4)


def test_validation_curve_sweeps_parameter_values():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression(lr=0.5, random_state=0)

    values, train_scores, test_scores = validation_curve(
        model,
        X,
        y,
        param_name="n_iter",
        param_range=[500, 1000, 3000],
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
    )

    assert values.tolist() == [500, 1000, 3000]
    assert train_scores.shape == (3, 3)
    assert test_scores.shape == (3, 3)
