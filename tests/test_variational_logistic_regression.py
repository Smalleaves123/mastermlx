import numpy as np

from mastermlx.variational import VariationalLogisticRegression


def test_variational_logistic_regression_separates_simple_data():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    model = VariationalLogisticRegression(alpha=1.0, max_iter=300, tol=1e-6).fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert model.score(X, y) >= 0.75


def test_variational_logistic_regression_predict_proba_sums_to_one():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    model = VariationalLogisticRegression(max_iter=300).fit(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_variational_logistic_regression_returns_scalar_prediction_for_single_sample():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    model = VariationalLogisticRegression(max_iter=300).fit(X, y)
    pred = model.predict([2.5])
    score = model.decision_function([2.5])

    assert pred in {0, 1}
    assert isinstance(score, float)


def test_variational_logistic_regression_samples_posterior_weights_and_probabilities():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    model = VariationalLogisticRegression(max_iter=300).fit(X, y)
    weights = model.sample_posterior_weights(n_samples=6, random_state=0)
    probs = model.sample_posterior_predict_proba([[2.0]], n_samples=5, random_state=0)

    assert weights.shape == (6, 2)
    assert probs.shape == (5, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
