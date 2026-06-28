import numpy as np

from mastermlx.neural_net import MLPClassifier, MLPRegressor


def test_mlp_classifier_solves_xor():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = np.array([0, 1, 1, 0])

    model = MLPClassifier(hidden_layer_sizes=(4,), activation="tanh", lr=0.5, n_iter=5000, random_state=0)
    model.fit(X, y)
    pred = model.predict(X)

    assert np.array_equal(pred, y)
    assert model.score(X, y) == 1.0


def test_mlp_regressor_fits_simple_line():
    X = np.arange(10, dtype=float).reshape(-1, 1)
    y = 2.0 * X.ravel() - 1.0

    model = MLPRegressor(hidden_layer_sizes=(), activation="tanh", optimizer="adam", lr=0.01, n_iter=2000, random_state=0)
    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert np.allclose(pred, y, atol=5e-2)
    assert model.score(X, y) > 0.99
