import numpy as np

from mastermlx.trees import GradientBoostingClassifier


def test_gradient_boosting_classifier_fits_simple_binary_boundary():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [0.2, 0.1],
        [0.8, 0.7],
        [1.0, 1.0],
        [1.0, 0.2],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_depth=2)
    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert model.score(X, y) >= 0.83


def test_gradient_boosting_classifier_predict_proba_sums_to_one():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.2, 0.2],
        [0.8, 0.9],
    ])
    y = np.array([0, 1, 2, 2, 0, 1])

    model = GradientBoostingClassifier(n_estimators=15, learning_rate=0.1, max_depth=2)
    model.fit(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (6, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_gradient_boosting_classifier_single_sample_predict():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=1)
    model.fit(X, y)

    pred = model.predict([2.5])
    assert pred in {0, 1}


def test_gradient_boosting_classifier_staged_interfaces_progress():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [0.2, 0.1],
        [0.8, 0.7],
        [1.0, 1.0],
        [1.0, 0.2],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = GradientBoostingClassifier(n_estimators=8, learning_rate=0.1, max_depth=2)
    model.fit(X, y)

    staged_proba = list(model.staged_predict_proba(X))
    staged_pred = list(model.staged_predict(X))

    assert len(staged_proba) == 8
    assert len(staged_pred) == 8
    assert np.allclose(staged_proba[-1], model.predict_proba(X))
    assert np.array_equal(staged_pred[-1], model.predict(X))
