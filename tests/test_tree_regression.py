import numpy as np

from mastermlx.trees import DecisionTreeRegressor, RandomForestRegressor


def test_decision_tree_regressor_fits_piecewise_data():
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y = np.array([0.0, 0.0, 0.5, 0.5, 2.0, 2.0, 3.0, 3.0])

    model = DecisionTreeRegressor(max_depth=3)
    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert model.score(X, y) > 0.95


def test_random_forest_regressor_fits_simple_curve():
    X = np.linspace(0.0, 1.0, 12).reshape(-1, 1)
    y = X.ravel() ** 2

    model = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=0)
    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert model.score(X, y) > 0.8
