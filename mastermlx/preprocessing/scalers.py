from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..base import BaseTransformer
from ..utils.validation import check_X


class StandardScaler(BaseTransformer):
    """Scale features to zero mean and unit variance."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "StandardScaler":
        X = check_X(X, dtype=float)
        self._set_n_features(X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_ = np.where(self.scale_ == 0.0, 1.0, self.scale_)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        self._check_fitted(["mean_", "scale_"])
        X = self._check_X(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X: ArrayLike) -> np.ndarray:
        self._check_fitted(["mean_", "scale_"])
        X = self._check_X(X, dtype=float)
        return X * self.scale_ + self.mean_


class MinMaxScaler(BaseTransformer):
    """Scale features into a fixed range."""

    def __init__(self, feature_range=(0.0, 1.0)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, X, y=None):
        X = check_X(X, dtype=float)
        self._set_n_features(X)
        low, high = self.feature_range
        if high <= low:
            raise ValueError("feature_range must have high > low")
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        span = self.data_max_ - self.data_min_
        span = np.where(span == 0.0, 1.0, span)
        self.scale_ = (high - low) / span
        self.min_ = low - self.data_min_ * self.scale_
        return self

    def transform(self, X):
        self._check_fitted(["min_", "scale_"])
        X = self._check_X(X, dtype=float)
        return X * self.scale_ + self.min_

    def inverse_transform(self, X):
        self._check_fitted(["min_", "scale_"])
        X = self._check_X(X, dtype=float)
        return (X - self.min_) / self.scale_


class MaxAbsScaler(BaseTransformer):
    """Scale each feature by its max absolute value."""

    def __init__(self):
        self.scale_ = None

    def fit(self, X, y=None):
        X = check_X(X, dtype=float)
        self._set_n_features(X)
        self.scale_ = np.max(np.abs(X), axis=0)
        self.scale_ = np.where(self.scale_ == 0.0, 1.0, self.scale_)
        return self

    def transform(self, X):
        self._check_fitted("scale_")
        X = self._check_X(X, dtype=float)
        return X / self.scale_

    def inverse_transform(self, X):
        self._check_fitted("scale_")
        X = self._check_X(X, dtype=float)
        return X * self.scale_


class RobustScaler(BaseTransformer):
    """Scale features using median and interquartile range."""

    def __init__(self, quantile_range=(25.0, 75.0)):
        self.quantile_range = quantile_range
        self.center_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        X = check_X(X, dtype=float)
        self._set_n_features(X)
        q_low, q_high = self.quantile_range
        if not 0.0 <= q_low < q_high <= 100.0:
            raise ValueError("quantile_range must satisfy 0 <= low < high <= 100")
        self.center_ = np.median(X, axis=0)
        low = np.percentile(X, q_low, axis=0)
        high = np.percentile(X, q_high, axis=0)
        self.scale_ = high - low
        self.scale_ = np.where(self.scale_ == 0.0, 1.0, self.scale_)
        return self

    def transform(self, X):
        self._check_fitted(["center_", "scale_"])
        X = self._check_X(X, dtype=float)
        return (X - self.center_) / self.scale_

    def inverse_transform(self, X):
        self._check_fitted(["center_", "scale_"])
        X = self._check_X(X, dtype=float)
        return X * self.scale_ + self.center_
