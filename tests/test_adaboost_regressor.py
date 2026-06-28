import numpy as np

from mastermlx.trees import AdaBoostRegressor


def test_adaboost_regressor_fits_simple_line():
    X = np.linspace(0.0, 2.0, 20).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 1.0

    model = AdaBoostRegressor(n_estimators=15, learning_rate=0.5)
    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert model.score(X, y) >= 0.8


def test_adaboost_regressor_single_sample_predict():
    X = np.linspace(0.0, 1.0, 10).reshape(-1, 1)
    y = X[:, 0] ** 2

    model = AdaBoostRegressor(n_estimators=10, learning_rate=0.5)
    model.fit(X, y)

    pred = model.predict([0.5])
    assert isinstance(pred, float)
