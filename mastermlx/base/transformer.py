from __future__ import annotations

from abc import ABC, abstractmethod

from .common import BaseAPI
from ..utils.estimator import get_params as _get_params, set_params as _set_params


class BaseTransformer(BaseAPI, ABC):
    """Common interface for preprocessing transforms."""

    @abstractmethod
    def fit(self, X, y=None):
        raise NotImplementedError

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_params(self, deep=True):
        return _get_params(self, deep=deep)

    def set_params(self, **params):
        return _set_params(self, **params)
