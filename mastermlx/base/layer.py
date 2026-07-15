from __future__ import annotations

from abc import ABC, abstractmethod

from ..utils.estimator import get_params as _get_params, set_params as _set_params
from .module import Module


class BaseLayer(Module, ABC):
    """Base interface for neural network layers."""

    @abstractmethod
    def forward(self, X):
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad):
        raise NotImplementedError

    def get_params(self, deep=True):
        return _get_params(self, deep=deep)

    def set_params(self, **params):
        return _set_params(self, **params)
