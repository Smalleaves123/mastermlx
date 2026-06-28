from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ..utils.validation import check_2d_array


class SelectFromModel(BaseTransformer):
    """Select features based on importance weights from a fitted model."""

    def __init__(self, estimator, threshold="mean", max_features=None):
        self.estimator = estimator
        self.threshold = threshold
        self.max_features = max_features
        self.support_ = None
        self.estimator_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X)
        if y is not None:
            self.estimator.fit(X, y)
        else:
            self.estimator.fit(X)
        self.estimator_ = self.estimator

        # Get feature importance
        if hasattr(self.estimator, 'coef_'):
            imp = np.abs(np.asarray(self.estimator.coef_)).ravel()
        elif hasattr(self.estimator, 'feature_importances_'):
            imp = np.asarray(self.estimator.feature_importances_).ravel()
        else:
            raise ValueError("estimator must have coef_ or feature_importances_")
        if imp.size != X.shape[1]:
            imp = imp[:X.shape[1]]

        # Compute threshold
        if self.threshold == "mean":
            t = float(np.mean(imp))
        elif self.threshold == "median":
            t = float(np.median(imp))
        elif isinstance(self.threshold, str) and self.threshold.endswith("*mean"):
            factor = float(self.threshold.replace("*mean", ""))
            t = factor * float(np.mean(imp))
        else:
            t = float(self.threshold)

        self.support_ = imp >= t

        # Max features cap
        if self.max_features is not None:
            k = min(int(self.max_features), X.shape[1])
            top_idx = np.argsort(imp)[-k:]
            keep = np.zeros(X.shape[1], dtype=bool)
            keep[top_idx] = True
            self.support_ = self.support_ & keep

        return self

    def transform(self, X):
        X = check_2d_array(X)
        if self.support_ is None:
            raise RuntimeError("Selector has not been fit yet")
        return X[:, self.support_]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
