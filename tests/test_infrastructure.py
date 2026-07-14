import numpy as np
import pytest

from mastermlx.linear_models import LinearRegression, LogisticRegression
from mastermlx.preprocessing import Pipeline, PolynomialFeatures, StandardScaler
from mastermlx.utils import (
    NotFittedError,
    check_X,
    check_X_y,
    check_is_fitted,
    clip_gradients,
    clone,
    resolve_rng,
    set_params,
    shuffle_indices,
)


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


def test_common_validation_tracks_features_and_fit_state():
    model = LinearRegression().fit([[0.0, 1.0], [1.0, 2.0]], [0.0, 1.0])

    assert model.n_features_in_ == 2
    assert check_X([[1.0, 2.0]]).shape == (1, 2)
    assert check_X_y([[1.0], [2.0]], [0.0, 1.0])[0].shape == (2, 1)
    check_is_fitted(model, ["coef_", "intercept_"])

    with pytest.raises(NotFittedError):
        LinearRegression().predict([[0.0, 1.0]])
    with pytest.raises(ValueError, match="different number of features"):
        model.predict([[0.0, 1.0, 2.0]])


def test_scaler_uses_common_fit_state_and_feature_check():
    scaler = StandardScaler()

    with pytest.raises(NotFittedError):
        scaler.transform([[1.0, 2.0]])

    scaler.fit([[1.0, 2.0], [3.0, 4.0]])
    assert scaler.n_features_in_ == 2
    with pytest.raises(ValueError, match="different number of features"):
        scaler.transform([[1.0, 2.0, 3.0]])


def test_resolve_rng_accepts_generator_and_preserves_reproducibility():
    rng = np.random.default_rng(7)
    assert resolve_rng(rng) is rng
    assert np.array_equal(
        shuffle_indices(8, random_state=np.random.default_rng(7)),
        shuffle_indices(8, random_state=np.random.default_rng(7)),
    )


def test_split_modules_keep_compatibility_facades():
    from mastermlx.math_tools import time_series as ts
    from mastermlx.math_tools.ts_core import rolling_mean as core_rolling_mean
    from mastermlx.neural_net import sequential as seq
    from mastermlx.neural_net.seq_fit import _SequentialFit

    assert ts.rolling_mean is core_rolling_mean
    assert seq.Sequential.__mro__[1].__name__ == "_SequentialRuntime"
    assert hasattr(_SequentialFit, "fit")


def test_estimator_state_can_be_saved_and_restored(tmp_path):
    model = LinearRegression().fit([[0.0], [1.0], [2.0]], [1.0, 3.0, 5.0])
    expected = model.predict([[3.0]])
    path = tmp_path / "linear.pkl"

    model.save(path)
    restored = LinearRegression.load(path)

    assert np.allclose(restored.predict([[3.0]]), expected)
    assert restored.state_dict()["n_features_in_"] == 1
