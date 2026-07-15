import inspect

import mastermlx
from mastermlx.base import BaseEstimator, Module
from mastermlx.neural_net import MLPClassifier, MLPRegressor, Sequential


def test_core_public_api_remains_available():
    for name in ("fit", "score", "save", "load", "get_params", "set_params"):
        assert hasattr(BaseEstimator, name)
    for name in ("state_dict", "load_state_dict", "save", "load", "save_checkpoint", "load_checkpoint"):
        assert hasattr(Module, name)
    for cls in (Sequential, MLPClassifier, MLPRegressor):
        assert hasattr(cls, "fit")
        assert hasattr(cls, "evaluate")
        assert "X" in inspect.signature(cls.fit).parameters
    assert hasattr(mastermlx, "get_backend")
    assert hasattr(mastermlx, "set_backend")


def test_checkpoint_signatures_are_stable():
    assert "path" in inspect.signature(Module.save_checkpoint).parameters
    assert "path" in inspect.signature(Module.load_checkpoint).parameters
    assert "strict" in inspect.signature(Module.load).parameters
