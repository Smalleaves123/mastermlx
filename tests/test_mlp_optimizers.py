import numpy as np

from mastermlx.neural_net import MLPClassifier, MLPRegressor, StepLR


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


def test_mlp_regressor_steps_learning_rate_scheduler():
    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = 2.0 * X.ravel() + 1.0

    model = MLPRegressor(
        hidden_layer_sizes=(),
        activation="relu",
        optimizer="sgd",
        lr=0.1,
        lr_scheduler=lambda optimizer: StepLR(optimizer, step_size=1, gamma=0.5),
        n_iter=3,
        tol=0.0,
        random_state=0,
    )
    model.fit(X, y)

    assert np.isclose(model.optimizer_.lr, 0.0125)


def test_mlp_regressor_rebinds_scheduler_to_current_optimizer():
    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = 2.0 * X.ravel() + 1.0

    scheduler = StepLR(None, step_size=1, gamma=0.5)
    model = MLPRegressor(
        hidden_layer_sizes=(),
        activation="relu",
        optimizer="sgd",
        lr=0.1,
        lr_scheduler=scheduler,
        n_iter=1,
        tol=0.0,
        random_state=0,
    )

    model.fit(X, y)
    first_optimizer = model.optimizer_
    assert scheduler.optimizer is first_optimizer
    assert np.isclose(first_optimizer.lr, 0.05)

    model.fit(X, y)
    assert scheduler.optimizer is model.optimizer_
    assert model.optimizer_ is not first_optimizer
    assert np.isclose(model.optimizer_.lr, 0.05)
