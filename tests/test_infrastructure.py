import numpy as np

from mastermlx.linear_models import LogisticRegression
from mastermlx.preprocessing import Pipeline, PolynomialFeatures, StandardScaler
from mastermlx.utils import clone, get_params, set_params, clip_gradients


class _Layer:
    def __init__(self):
        self.dW_ = np.array([3.0, 4.0])
        self.db_ = np.array([0.0, 0.0])


def test_base_estimator_params_round_trip():
    model = LogisticRegression(lr=0.2, n_iter=500, random_state=7)

    params = model.get_params()
    assert params["lr"] == 0.2
    assert params["n_iter"] == 500

    model.set_params(lr=0.05)
    assert model.lr == 0.05


def test_pipeline_get_params_exposes_nested_steps():
    pipe = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale", StandardScaler()),
        ]
    )

    params = pipe.get_params()
    assert params["poly__degree"] == 2
    assert "scale" in params
    assert not any(key.startswith("scale__") for key in params)

    pipe.set_params(poly__degree=3)
    assert pipe.steps[0][1].degree == 3


def test_clone_produces_independent_copy():
    pipe = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False))])
    cloned = clone(pipe)

    cloned.set_params(poly__degree=4)

    assert pipe.steps[0][1].degree == 2
    assert cloned.steps[0][1].degree == 4


def test_set_params_helper_supports_nested_objects():
    model = LogisticRegression(lr=0.1, n_iter=1000)
    set_params(model, lr=0.3, n_iter=2000)

    assert model.lr == 0.3
    assert model.n_iter == 2000


def test_clip_gradients_scales_large_updates():
    layer = _Layer()
    total_norm, scale = clip_gradients([layer], max_norm=2.5)

    assert total_norm > 2.5
    assert scale < 1.0
    assert np.isclose(np.sqrt(np.sum(layer.dW_ ** 2) + np.sum(layer.db_ ** 2)), 2.5)
