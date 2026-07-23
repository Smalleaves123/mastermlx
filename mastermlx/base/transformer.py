from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar

import numpy as np
from numpy.typing import ArrayLike

from .common import BaseAPI
from ..utils.estimator import get_params as _get_params, set_params as _set_params


TransformerT = TypeVar("TransformerT", bound="BaseTransformer")


class BaseTransformer(BaseAPI, ABC):
    """Common interface for preprocessing transforms."""

    @abstractmethod
    def fit(self: TransformerT, X: ArrayLike, y: ArrayLike | None = None) -> TransformerT:
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, X: ArrayLike, y: ArrayLike | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return _get_params(self, deep=deep)

    def set_params(self: TransformerT, **params: Any) -> TransformerT:
        return _set_params(self, **params)
