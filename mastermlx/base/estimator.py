from __future__ import annotations

from abc import ABC, abstractmethod
import copy

from .common import BaseAPI
from .checkpoint import load_object, save_object
from ..version import __version__
from ..utils.estimator import get_params as _get_params, set_params as _set_params


class BaseEstimator(BaseAPI, ABC):
    """Common interface for models with fit/predict semantics."""

    @abstractmethod
    def fit(self, X, y=None):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError("Subclasses must implement score() for their task type")

    def get_params(self, deep=True):
        return _get_params(self, deep=deep)

    def set_params(self, **params):
        return _set_params(self, **params)

    def state_dict(self):
        """Return a detached copy of the estimator state."""

        return copy.deepcopy(vars(self))

    def load_state_dict(self, state):
        """Restore state previously returned by :meth:`state_dict`."""

        if not isinstance(state, dict):
            raise TypeError("state must be a dict")
        self.__dict__.update(copy.deepcopy(state))
        return self

    def save(self, path):
        """Save the fitted estimator to a versioned safe archive."""

        save_object(self, path, __version__)
        return self

    @classmethod
    def load(cls, path):
        """Load an estimator saved with :meth:`save`."""
        obj = load_object(path)
        if not isinstance(obj, cls):
            raise TypeError(f"saved object must be a {cls.__name__}")
        for callback in getattr(obj, "callbacks", ()):
            if hasattr(callback, "set_model"):
                callback.set_model(obj)
        return obj
