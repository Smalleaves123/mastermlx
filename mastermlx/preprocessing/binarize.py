from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ..utils.validation import check_2d_array


class Binarizer(BaseTransformer):
    """Threshold features: values > thresh become 1, else 0."""

    def __init__(self, threshold=0.0):
        self.threshold = float(threshold)

    def fit(self, X, y=None):
        check_2d_array(X)
        return self

    def transform(self, X):
        X = check_2d_array(X).astype(float)
        return (X > self.threshold).astype(float)
