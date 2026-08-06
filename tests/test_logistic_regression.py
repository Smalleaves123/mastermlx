import numpy as np

from mastermlx.linear_models import LogisticRegression


def test_logistic_regression_separates_simple_data(linear_binary_data):
    X, y = linear_binary_data

    model = LogisticRegression(lr=0.5, n_iter=5000, random_state=0)
    model.fit(X, y)

    pred = model.predict(X)

    assert np.array_equal(pred, y)
    assert model.score(X, y) == 1.0


def test_logistic_regression_multiclass_keeps_probabilities_stable():
    X = np.array([
        [-3.0, -2.0], [-2.5, -1.5],
        [0.0, 3.0], [0.5, 2.5],
        [3.0, -1.0], [2.5, -0.5],
    ])
    y = np.array([10, 10, 20, 20, 30, 30])

    model = LogisticRegression(lr=0.2, n_iter=2000, random_state=0)
    model.fit(X, y)

    probabilities = model.predict_proba(X)
    assert np.isfinite(model.loss_).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.array_equal(model.predict(X), y)
    assert model.n_iter_ <= model.n_iter


def test_logistic_regression_rejects_invalid_training_parameters():
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])

    for kwargs in (
        {"lr": 0.0},
        {"n_iter": 0},
        {"batch_size": 0},
        {"tol": -1.0},
    ):
        model = LogisticRegression(**kwargs)
        try:
            model.fit(X, y)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")
