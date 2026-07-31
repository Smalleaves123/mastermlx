import numpy as np

from mastermlx.linear_models import LogisticRegression


def test_logistic_regression_separates_simple_data(linear_binary_data):
    X, y = linear_binary_data

    model = LogisticRegression(lr=0.5, n_iter=5000, random_state=0)
    model.fit(X, y)

    pred = model.predict(X)

    assert np.array_equal(pred, y)
    assert model.score(X, y) == 1.0
