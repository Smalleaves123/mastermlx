from __future__ import annotations

from itertools import combinations, combinations_with_replacement

import numpy as np

from ..base import BaseTransformer
from ..utils.validation import check_2d_array


class PolynomialFeatures(BaseTransformer):
    """Generate polynomial and interaction features."""

    def __init__(self, degree=2, include_bias=True, interaction_only=False):
        self.degree = int(degree)
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.powers_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        if self.degree < 1:
            raise ValueError("degree must be at least 1")

        n_features = X.shape[1]
        combos = []
        if self.include_bias:
            combos.append(np.zeros(n_features, dtype=int))

        choose = combinations if self.interaction_only else combinations_with_replacement
        for deg in range(1, self.degree + 1):
            for idx in choose(range(n_features), deg):
                power = np.zeros(n_features, dtype=int)
                for item in idx:
                    power[item] += 1
                combos.append(power)

        self.powers_ = np.asarray(combos, dtype=int)
        self.n_features_in_ = n_features
        return self

    def transform(self, X):
        X = check_2d_array(X).astype(float)
        if self.powers_ is None:
            raise RuntimeError("PolynomialFeatures has not been fit yet")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the fitted data")

        out = []
        for power in self.powers_:
            term = np.ones(X.shape[0], dtype=float)
            for j, p in enumerate(power):
                if p:
                    term *= X[:, j] ** p
            out.append(term)
        return np.column_stack(out)


PolyFeatures = PolynomialFeatures
