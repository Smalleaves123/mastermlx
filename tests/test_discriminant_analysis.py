import numpy as np

from mastermlx.probabilistic import LDA, QDA


def test_lda_fits_simple_gaussian_classes():
    X = np.array([
        [0.0, 0.0],
        [0.1, -0.1],
        [1.0, 1.0],
        [1.1, 0.9],
    ])
    y = np.array([0, 0, 1, 1])

    model = LDA()
    pred = model.fit(X, y).predict(X)

    assert np.array_equal(pred, y)
    assert model.score(X, y) == 1.0


def test_qda_fits_simple_gaussian_classes():
    X = np.array([
        [0.0, 0.0],
        [0.2, -0.1],
        [2.0, 2.0],
        [2.2, 1.9],
    ])
    y = np.array([0, 0, 1, 1])

    model = QDA()
    pred = model.fit(X, y).predict(X)

    assert np.array_equal(pred, y)
    assert model.score(X, y) == 1.0
