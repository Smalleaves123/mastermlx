import numpy as np

from mastermlx.linear_models import LinearRegression


def test_linear_regression_fits_line():
    X = np.arange(10).reshape(-1, 1)
    y = 3.0 * X.ravel() + 2.0

    model = LinearRegression()
    model.fit(X, y)

    preds = model.predict(X)

    assert np.allclose(model.intercept_, 2.0, atol=1e-8)
    assert np.allclose(model.coef_, np.array([3.0]), atol=1e-8)
    assert np.allclose(preds, y, atol=1e-8)
    assert np.isclose(model.score(X, y), 1.0, atol=1e-8)

