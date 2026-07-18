
import numpy as np
import pytest

from mastermlx.base import Module, Parameter
from mastermlx.neural_net import BatchNorm, Dense, Dropout, ReLU, Sequential


def test_sequential_exposes_parameters_and_roundtrips_state(tmp_path):
    model = Sequential([Dense(3, 4, random_state=0), ReLU(), Dense(4, 2, random_state=1)])
    state = model.state_dict()
    assert isinstance(model, Module)
    assert all(isinstance(parameter, Parameter) for parameter in model.parameters())
    assert set(state) == {"layers.0.W", "layers.0.b", "layers.2.W", "layers.2.b"}

    expected = {key: value.copy() for key, value in state.items()}
    model.layers[0].W_[:] = 0.0
    result = model.load_state_dict(expected)
    assert result["missing"] == []
    assert result["unexpected"] == []
    assert np.allclose(model.state_dict()["layers.0.W"], expected["layers.0.W"])

    path = model.save(tmp_path / "model")
    model.layers[2].b_[:] = 1.0
    model.load(path)
    assert np.allclose(model.layers[2].b_, expected["layers.2.b"])


def test_module_mode_propagates_to_batch_norm_and_dropout():
    model = Sequential([Dense(3, 3, random_state=0), BatchNorm(3), Dropout(0.5), ReLU()])
    model.eval()
    assert all(getattr(layer, "training", False) is False for layer in model.layers)
    model.train()
    assert all(getattr(layer, "training", True) is True for layer in model.layers)


def test_state_loading_reports_shape_and_key_errors():
    model = Sequential([Dense(2, 2, random_state=0)])
    state = model.state_dict()
    with pytest.raises(ValueError, match="state mismatch"):
        model.load_state_dict({"layers.0.W": state["layers.0.W"]})
    bad = dict(state)
    bad["layers.0.W"] = np.zeros((3, 3))
    with pytest.raises(ValueError, match="shape"):
        model.load_state_dict(bad)
