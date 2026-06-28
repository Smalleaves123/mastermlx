import numpy as np

from mastermlx.svm import KernelSVR


def test_kernel_svr_fits_curved_signal():
    X = np.linspace(-2.0, 2.0, 30).reshape(-1, 1)
    y = np.sin(X.ravel())

    model = KernelSVR(C=3.0, epsilon=0.05, kernel="rbf", gamma=1.5, lr=0.05, max_iter=3000)
    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert model.score(X, y) > 0.9


def test_kernel_svr_predicts_single_value():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 0.0, -1.0])

    model = KernelSVR(C=2.0, epsilon=0.1, kernel="poly", degree=2, coef0=1.0, lr=0.03, max_iter=2500).fit(X, y)
    pred = model.predict([1.5])

    assert isinstance(pred, float)
