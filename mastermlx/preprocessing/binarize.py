from __future__ import annotations


from ..base import BaseTransformer
from ..utils.validation import check_2d_array


class Binarizer(BaseTransformer):
    """Threshold features: values > thresh become 1, else 0."""

    def __init__(self, threshold=0.0):
        self.threshold = float(threshold)

    def fit(self, X, y=None):
        X = check_2d_array(X)
        self._set_n_features(X)
        return self

    def transform(self, X):
        X = self._check_X(X, dtype=float)
        return (X > self.threshold).astype(float)
