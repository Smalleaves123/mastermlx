import numpy as np

from mastermlx.data import GridSearchCV, GroupKFold, KFold, RandomizedSearchCV
from mastermlx.linear_models import LogisticRegression
from mastermlx.preprocessing import Pipeline, PolynomialFeatures, StandardScaler


class _BareEstimator:
    def __init__(self, bias=0.0):
        self.bias = bias

    def fit(self, X, y):
        self.mean_ = float(np.mean(y)) + self.bias
        return self

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], self.mean_)

    def score(self, X, y):
        return -float(np.mean((self.predict(X) - y) ** 2))


def test_grid_search_finds_best_estimator():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    search = GridSearchCV(
        LogisticRegression(random_state=0),
        param_grid={"lr": [0.05, 0.2], "n_iter": [500, 3000]},
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
    )
    search.fit(X, y)

    assert search.best_estimator_ is not None
    assert search.best_params_["n_iter"] in {500, 3000}
    assert search.best_score_ >= 0.5
    assert set(search.cv_results_) == {
        "params",
        "mean_test_score",
        "std_test_score",
        "mean_fit_time",
        "mean_score_time",
    }


def test_grid_search_supports_pipeline_params():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    pipe = Pipeline(
        [
            ("poly", PolynomialFeatures(include_bias=False)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(random_state=0)),
        ]
    )
    search = GridSearchCV(
        pipe,
        param_grid={"poly__degree": [1, 2], "clf__n_iter": [1000, 3000], "clf__lr": [0.1, 0.5]},
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        return_train_score=True,
    )
    search.fit(X, y)
    pred = search.predict(X)

    assert pred.shape == y.shape
    assert "mean_train_score" in search.cv_results_
    assert search.score(X, y) >= 0.8


def test_grid_search_supports_multiple_metrics():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    search = GridSearchCV(
        LogisticRegression(random_state=0),
        param_grid={"lr": [0.05, 0.2], "n_iter": [500, 3000]},
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        scoring={"acc": "accuracy", "mse": lambda yt, yp: -np.mean((yt - yp) ** 2)},
        refit="acc",
        return_train_score=True,
    )
    search.fit(X, y)

    assert "mean_test_acc" in search.cv_results_
    assert "mean_test_mse" in search.cv_results_
    assert "mean_train_acc" in search.cv_results_
    assert search.best_estimator_ is not None
    assert search.best_score_ >= 0.5


def test_randomized_search_finds_estimator_with_reproducible_sampling():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    search_a = RandomizedSearchCV(
        LogisticRegression(random_state=0),
        param_distributions={"lr": [0.05, 0.1, 0.2], "n_iter": [500, 1000, 3000]},
        n_iter=4,
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        random_state=7,
    ).fit(X, y)
    search_b = RandomizedSearchCV(
        LogisticRegression(random_state=0),
        param_distributions={"lr": [0.05, 0.1, 0.2], "n_iter": [500, 1000, 3000]},
        n_iter=4,
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        random_state=7,
    ).fit(X, y)

    assert search_a.best_estimator_ is not None
    assert search_a.best_params_ == search_b.best_params_
    assert len(search_a.cv_results_["params"]) == 4


def test_randomized_search_supports_pipeline_and_multi_metric():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    pipe = Pipeline(
        [
            ("poly", PolynomialFeatures(include_bias=False)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(random_state=0)),
        ]
    )
    search = RandomizedSearchCV(
        pipe,
        param_distributions={"poly__degree": [1, 2], "clf__n_iter": [1000, 3000], "clf__lr": [0.1, 0.5]},
        n_iter=5,
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        scoring={"acc": "accuracy", "mse": lambda yt, yp: -np.mean((yt - yp) ** 2)},
        refit="acc",
        return_train_score=True,
        random_state=3,
    )
    search.fit(X, y)

    assert "mean_test_acc" in search.cv_results_
    assert "mean_train_acc" in search.cv_results_
    assert search.best_estimator_ is not None
    assert search.score(X, y) >= 0.8


def test_grid_search_accepts_groups():
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])

    search = GridSearchCV(
        LogisticRegression(random_state=0),
        param_grid={"lr": [0.1, 0.5], "n_iter": [1000, 2000]},
        cv=GroupKFold(n_splits=2),
    )
    search.fit(X, y, groups=groups)

    assert search.best_estimator_ is not None
    assert len(search.cv_results_["params"]) == 4


def test_grid_search_applies_params_to_plain_estimators():
    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = np.arange(6, dtype=float)
    search = GridSearchCV(
        _BareEstimator(),
        param_grid={"bias": [-1.0, 0.0, 1.0]},
        cv=KFold(n_splits=3),
    ).fit(X, y)

    assert search.best_estimator_ is not None
    assert search.best_params_["bias"] == 0.0
