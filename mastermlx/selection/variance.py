from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ..utils import check_2d_array


class VarianceThreshold(BaseTransformer):
    def __init__(self, threshold=0.0):
        self.threshold = float(threshold)
        self.variances_ = None
        self.support_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        self.variances_ = np.var(X, axis=0)
        self.support_ = self.variances_ > self.threshold
        if not np.any(self.support_):
            raise ValueError("No feature meets the variance threshold")
        return self

    def transform(self, X):
        X = check_2d_array(X).astype(float)
        if self.support_ is None:
            raise RuntimeError("VarianceThreshold has not been fit yet")
        return X[:, self.support_]

    def get_support(self):
        if self.support_ is None:
            raise RuntimeError("VarianceThreshold has not been fit yet")
        return self.support_.copy()
