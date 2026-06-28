import numpy as np

from mastermlx.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from mastermlx.linear_models import LinearRegression, LogisticRegression
from mastermlx.trees import DecisionTreeClassifier, DecisionTreeRegressor


def _cls_data():
    X = np.array(
        [
            [0.0, 0.1],
            [0.2, 0.0],
            [0.1, 0.3],
            [2.8, 2.7],
            [3.0, 3.1],
            [3.2, 2.9],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


def _reg_data():
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 0.5],
            [4.0, 1.5],
            [5.0, 1.0],
        ]
    )
    y = 2.0 * X[:, 0] - 0.5 * X[:, 1] + 1.0
    return X, y


def test_bagging_models_work():
    X, y = _cls_data()
    clf = BaggingClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=5, random_state=0).fit(X, y)
    assert np.array_equal(clf.predict(X), y)
    assert clf.score(X, y) == 1.0

    Xr, yr = _reg_data()
    reg = BaggingRegressor(
        DecisionTreeRegressor(max_depth=None),
        n_estimators=5,
        bootstrap=False,
        max_samples=1.0,
        max_features=1.0,
        random_state=0,
    ).fit(Xr, yr)
    pred = reg.predict(Xr)
    assert pred.shape == yr.shape
    assert np.allclose(pred, yr, atol=1e-6)


def test_extra_trees_models_work():
    X, y = _cls_data()
    clf = ExtraTreesClassifier(n_estimators=8, max_depth=4, random_state=0).fit(X, y)
    assert np.array_equal(clf.predict(X), y)
    assert clf.score(X, y) == 1.0

    Xr, yr = _reg_data()
    reg = ExtraTreesRegressor(n_estimators=8, max_depth=4, random_state=0).fit(Xr, yr)
    pred = reg.predict(Xr)
    assert pred.shape == yr.shape
    assert np.mean((pred - yr) ** 2) < 0.25


def test_voting_models_work():
    X, y = _cls_data()
    clf = VotingClassifier(
        [
            LogisticRegression(lr=0.2, n_iter=400, random_state=0),
            LogisticRegression(lr=0.2, n_iter=400, random_state=1),
        ],
        voting="soft",
    ).fit(X, y)

    assert np.array_equal(clf.predict(X), y)
    assert np.allclose(clf.predict_proba(X).sum(axis=1), 1.0)

    Xr, yr = _reg_data()
    reg = VotingRegressor(
        [
            LinearRegression(),
            LinearRegression(),
        ]
    ).fit(Xr, yr)
    pred = reg.predict(Xr)
    assert np.allclose(pred, yr, atol=1e-8)


def test_stacking_models_work():
    X, y = _cls_data()
    clf = StackingClassifier(
        [
            LogisticRegression(lr=0.2, n_iter=400, random_state=0),
            DecisionTreeClassifier(max_depth=2),
        ],
        final_estimator=LogisticRegression(lr=0.2, n_iter=400, random_state=2),
        cv=3,
        use_proba=True,
    ).fit(X, y)
    assert np.array_equal(clf.predict(X), y)

    Xr, yr = _reg_data()
    reg = StackingRegressor(
        [
            LinearRegression(),
            DecisionTreeRegressor(max_depth=2),
        ],
        final_estimator=LinearRegression(),
        cv=3,
    ).fit(Xr, yr)
    pred = reg.predict(Xr)
    assert pred.shape == yr.shape
    assert np.allclose(pred, yr, atol=1e-6)
