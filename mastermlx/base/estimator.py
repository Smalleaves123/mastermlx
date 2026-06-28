from __future__ import annotations

from abc import ABC, abstractmethod

from ..utils.estimator import get_params as _get_params, set_params as _set_params


class BaseEstimator(ABC):
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
