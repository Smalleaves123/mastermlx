import numpy as np

from mastermlx.trees import AdaBoostClassifier, GradientBoostingRegressor


def test_adaboost_classifier_fits_simple_boundary():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [0.2, 0.1],
        [0.8, 0.7],
        [1.0, 1.0],
        [1.0, 0.2],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = AdaBoostClassifier(n_estimators=20, learning_rate=0.8)
    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert model.score(X, y) >= 0.83


def test_adaboost_classifier_single_sample_predict():
    X = np.array([[0.0], [0.5], [1.0], [1.5], [2.0]])
    y = np.array([0, 0, 1, 1, 1])

    model = AdaBoostClassifier(n_estimators=10)
    model.fit(X, y)

    pred = model.predict([1.8])
    assert pred == 1


def test_adaboost_classifier_staged_interfaces_progress():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [0.2, 0.1],
        [0.8, 0.7],
        [1.0, 1.0],
        [1.0, 0.2],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = AdaBoostClassifier(n_estimators=8, learning_rate=0.8)
    model.fit(X, y)

    staged_scores = list(model.staged_decision_function(X))
    staged_proba = list(model.staged_predict_proba(X))
    staged_pred = list(model.staged_predict(X))

    assert len(staged_scores) == len(model.estimators_)
    assert len(staged_proba) == len(model.estimators_)
    assert len(staged_pred) == len(model.estimators_)
    assert staged_pred[-1].shape == y.shape


def test_gradient_boosting_regressor_fits_curved_signal():
    X = np.linspace(-1.0, 1.0, 60).reshape(-1, 1)
    y = X[:, 0] ** 2 + 0.2 * X[:, 0]

    model = GradientBoostingRegressor(n_estimators=40, learning_rate=0.1, max_depth=2)
    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert model.score(X, y) >= 0.9


def test_gradient_boosting_regressor_single_sample_predict():
    X = np.linspace(0.0, 2.0, 20).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 1.0

    model = GradientBoostingRegressor(n_estimators=25, learning_rate=0.1, max_depth=1)
    model.fit(X, y)

    pred = model.predict([1.5])
    assert isinstance(pred, float)
