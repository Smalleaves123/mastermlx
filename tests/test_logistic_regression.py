import numpy as np

from mastermlx.linear_models import LogisticRegression


def test_logistic_regression_separates_simple_data():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(lr=0.5, n_iter=5000, random_state=0)
    model.fit(X, y)

    pred = model.predict(X)

    assert np.array_equal(pred, y)
    assert model.score(X, y) == 1.0

