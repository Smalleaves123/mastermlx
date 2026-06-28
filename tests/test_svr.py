import numpy as np

from mastermlx.svm import LinearSVR


def test_linear_svr_fits_simple_line():
    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = 2.5 * X.ravel() - 1.0

    model = LinearSVR(C=2.0, epsilon=0.05, lr=0.05, max_iter=5000, random_state=0)
    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert model.score(X, y) > 0.98


def test_linear_svr_predicts_single_value():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])

    model = LinearSVR(C=1.0, epsilon=0.1, lr=0.05, max_iter=3000, random_state=0).fit(X, y)
    pred = model.predict([1.5])
    assert isinstance(pred, float)
