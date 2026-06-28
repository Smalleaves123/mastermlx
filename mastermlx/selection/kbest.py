from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ..utils import check_2d_array, check_1d_array


def f_classif(X, y):
    X = check_2d_array(X).astype(float)
    y = check_1d_array(y, name="y")
    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError("f_classif requires at least two classes")

    n_features = X.shape[1]
    scores = np.zeros(n_features, dtype=float)
    pvals = np.full(n_features, np.nan, dtype=float)
    overall = np.mean(X, axis=0)

    for j in range(n_features):
        ssb = 0.0
        ssw = 0.0
        for cls in classes:
            mask = y == cls
            x = X[mask, j]
            if x.size == 0:
                continue
            m = np.mean(x)
            ssb += x.size * (m - overall[j]) ** 2
            ssw += np.sum((x - m) ** 2)
        dfb = classes.size - 1
        dfw = X.shape[0] - classes.size
        if dfb <= 0 or dfw <= 0 or ssw <= 0:
            scores[j] = 0.0
        else:
            scores[j] = (ssb / dfb) / (ssw / dfw)
    return scores, pvals


def f_regression(X, y):
    X = check_2d_array(X).astype(float)
    y = check_1d_array(y, name="y").astype(float)
    n = X.shape[0]
    if n < 3:
        raise ValueError("f_regression requires at least 3 samples")

    y0 = y - np.mean(y)
    yss = np.sum(y0 ** 2)
    scores = np.zeros(X.shape[1], dtype=float)
    pvals = np.full(X.shape[1], np.nan, dtype=float)

    for j in range(X.shape[1]):
        x = X[:, j] - np.mean(X[:, j])
        xss = np.sum(x ** 2)
        if xss <= 0.0 or yss <= 0.0:
            scores[j] = 0.0
            continue
        r = np.sum(x * y0) / np.sqrt(xss * yss)
        r = np.clip(r, -1.0, 1.0)
        r2 = r ** 2
        if r2 >= 1.0:
            scores[j] = np.inf
        else:
            scores[j] = (r2 / (1.0 - r2)) * (n - 2)
    return scores, pvals


class SelectKBest(BaseTransformer):
    def __init__(self, score_func=f_classif, k=10):
        self.score_func = score_func
        self.k = k
        self.scores_ = None
        self.pvalues_ = None
        self.support_ = None
        self.ranking_ = None

    def _k(self, n_features):
        if self.k == "all":
            return n_features
        k = int(self.k)
        if k < 1:
            raise ValueError("k must be at least 1 or 'all'")
        return min(k, n_features)

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        if y is None:
            raise ValueError("SelectKBest requires y")
        scores, pvals = self.score_func(X, y)
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 1 or scores.shape[0] != X.shape[1]:
            raise ValueError("score_func must return one score per feature")
        self.scores_ = scores
        self.pvalues_ = None if pvals is None else np.asarray(pvals, dtype=float)

        k = self._k(X.shape[1])
        order = np.argsort(scores)[::-1]
        keep = order[:k]
        support = np.zeros(X.shape[1], dtype=bool)
        support[keep] = True
        self.support_ = support
        self.ranking_ = np.empty(X.shape[1], dtype=int)
        self.ranking_[order] = np.arange(1, X.shape[1] + 1)
        return self

    def transform(self, X):
        X = check_2d_array(X).astype(float)
        if self.support_ is None:
            raise RuntimeError("SelectKBest has not been fit yet")
        return X[:, self.support_]

    def get_support(self):
        if self.support_ is None:
            raise RuntimeError("SelectKBest has not been fit yet")
        return self.support_.copy()
