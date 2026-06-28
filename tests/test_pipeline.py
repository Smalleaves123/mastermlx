import numpy as np

from mastermlx.linear_models import LogisticRegression
from mastermlx.preprocessing import Pipeline, PolynomialFeatures, StandardScaler


def test_pipeline_fits_transformers_and_estimator():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0, 0, 0, 1])

    pipe = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(lr=0.1, n_iter=4000, random_state=0)),
            ]
        )
    pipe.fit(X, y)
    pred = pipe.predict(X)

    assert np.array_equal(pred, y)
    assert pipe.score(X, y) == 1.0


def test_pipeline_transform_returns_last_transformer_output():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    pipe = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False))]).fit(X)

    Z = pipe.transform(X)

    assert Z.shape == (2, 5)


def test_pipeline_set_params_updates_nested_step():
    pipe = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale", StandardScaler()),
        ]
    )

    pipe.set_params(poly__degree=3)

    assert pipe.steps[0][1].degree == 3


def test_pipeline_fit_transform_runs_transform_chain():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    pipe = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False))])

    Z = pipe.fit_transform(X)

    assert Z.shape == (2, 5)


def test_pipeline_predict_proba_uses_final_estimator():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(lr=0.5, n_iter=3000, random_state=0))]).fit(X, y)

    proba = pipe.predict_proba(X)

    assert proba.shape == (6, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-8)
