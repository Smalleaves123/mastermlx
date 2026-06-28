import numpy as np

from mastermlx.trees import DecisionTreeClassifier


def test_decision_tree_fits_simple_data():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = np.array([0, 0, 0, 1])

    model = DecisionTreeClassifier(max_depth=2)
    model.fit(X, y)
    pred = model.predict(X)

    assert np.array_equal(pred, y)
    assert model.score(X, y) == 1.0


def test_decision_tree_single_sample_predict():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    model = DecisionTreeClassifier(max_depth=1)
    model.fit(X, y)

    pred = model.predict([2.5])
    assert pred == 1

