from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from pathlib import Path
from typing import Any, TypeVar

from numpy.typing import ArrayLike

from .common import BaseAPI
from .checkpoint import load_object, save_object
from ..version import __version__
from ..utils.estimator import get_params as _get_params, set_params as _set_params


EstimatorT = TypeVar("EstimatorT", bound="BaseEstimator")


class BaseEstimator(BaseAPI, ABC):
    """Common interface for models with fit/predict semantics."""

    @abstractmethod
    def fit(self: EstimatorT, X: ArrayLike, y: ArrayLike | None = None) -> EstimatorT:
        raise NotImplementedError

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        raise NotImplementedError("Subclasses must implement score() for their task type")

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return _get_params(self, deep=deep)

    def set_params(self: EstimatorT, **params: Any) -> EstimatorT:
        return _set_params(self, **params)

    def state_dict(self) -> dict[str, Any]:
        """Return a detached copy of the estimator state."""

        return copy.deepcopy(vars(self))

    def load_state_dict(self: EstimatorT, state: dict[str, Any]) -> EstimatorT:
        """Restore state previously returned by :meth:`state_dict`."""

        if not isinstance(state, dict):
            raise TypeError("state must be a dict")
        self.__dict__.update(copy.deepcopy(state))
        return self

    def save(self: EstimatorT, path: str | Path) -> EstimatorT:
        """Save the fitted estimator to a versioned safe archive."""

        save_object(self, path, __version__)
        return self

    @classmethod
    def load(cls: type[EstimatorT], path: str | Path) -> EstimatorT:
        """Load an estimator saved with :meth:`save`."""
        obj = load_object(path)
        if not isinstance(obj, cls):
            raise TypeError(f"saved object must be a {cls.__name__}")
        for callback in getattr(obj, "callbacks", ()):
            if hasattr(callback, "set_model"):
                callback.set_model(obj)
        return obj
