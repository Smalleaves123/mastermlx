from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ..utils import check_1d_array, check_2d_array, clone


class RFE(BaseTransformer):
    def __init__(self, estimator, n_features_to_select=None, step=1):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.support_ = None
        self.ranking_ = None
        self.estimator_ = None

    def _n_select(self, n_features):
        if self.n_features_to_select is None:
            return max(1, n_features // 2)
        n = int(self.n_features_to_select)
        if n < 1 or n > n_features:
            raise ValueError("n_features_to_select must be between 1 and n_features")
        return n

    def _step(self, n_features):
        step = self.step
        if isinstance(step, float):
            if not 0.0 < step < 1.0:
                raise ValueError("step as float must be between 0 and 1")
            return max(1, int(np.ceil(step * n_features)))
        step = int(step)
        if step < 1:
            raise ValueError("step must be at least 1")
        return step

    def _importance(self, est):
        if hasattr(est, "coef_"):
            coef = np.asarray(est.coef_, dtype=float)
            if coef.ndim == 1:
                return np.abs(coef)
            return np.sum(np.abs(coef), axis=0)
        if hasattr(est, "feature_importances_"):
            return np.asarray(est.feature_importances_, dtype=float)
        raise ValueError("estimator must expose coef_ or feature_importances_ after fit")

    def fit(self, X, y):
        X = check_2d_array(X).astype(float)
        y = check_1d_array(y, name="y")
        n_features = X.shape[1]
        n_keep = self._n_select(n_features)
        active = np.ones(n_features, dtype=bool)
        ranking = np.ones(n_features, dtype=int)
        rank = 2

        while np.sum(active) > n_keep:
            est = clone(self.estimator)
            est.fit(X[:, active], y)
            imp = self._importance(est)
            if imp.shape[0] != np.sum(active):
                raise ValueError("importance vector has wrong shape")
            drop = min(self._step(np.sum(active)), np.sum(active) - n_keep)
            order = np.argsort(imp)
            remove_local = order[:drop]
            active_idx = np.flatnonzero(active)
            removed = active_idx[remove_local]
            active[removed] = False
            ranking[removed] = rank
            rank += 1

        self.support_ = active
        self.ranking_ = ranking
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[:, active], y)
        return self

    def transform(self, X):
        X = check_2d_array(X).astype(float)
        if self.support_ is None:
            raise RuntimeError("RFE has not been fit yet")
        return X[:, self.support_]

    def get_support(self):
        if self.support_ is None:
            raise RuntimeError("RFE has not been fit yet")
        return self.support_.copy()
