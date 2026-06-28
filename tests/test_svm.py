import numpy as np

from mastermlx.svm import SVC


def test_svc_fits_simple_linear_separation():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = np.array([0, 0, 0, 1])

    model = SVC(kernel="linear", C=10.0, max_iter=1000, random_state=0)
    model.fit(X, y)
    pred = model.predict(X)

    assert np.array_equal(pred, y)
    assert model.score(X, y) == 1.0


def test_svc_handles_multiclass_ovr():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [2.0, 2.0],
        [2.0, 3.0],
        [3.0, 2.0],
    ])
    y = np.array([0, 0, 0, 1, 1, 2])

    model = SVC(kernel="rbf", C=5.0, gamma=1.0, max_iter=2000, random_state=0)
    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert set(pred.tolist()) <= {0, 1, 2}
