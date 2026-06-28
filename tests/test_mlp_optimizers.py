import numpy as np

from mastermlx.neural_net import MLPClassifier, MLPRegressor


def test_mlp_classifier_with_adam_solves_xor():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = np.array([0, 1, 1, 0])

    model = MLPClassifier(hidden_layer_sizes=(4,), activation="tanh", optimizer="adam", lr=0.05, n_iter=3000, random_state=0)
    model.fit(X, y)

    assert np.array_equal(model.predict(X), y)


def test_mlp_regressor_with_rmsprop_fits_line():
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y = 3.0 * X.ravel() - 2.0

    model = MLPRegressor(hidden_layer_sizes=(6,), activation="relu", optimizer="rmsprop", lr=0.01, n_iter=3000, random_state=0)
    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert np.allclose(pred, y, atol=0.5)
