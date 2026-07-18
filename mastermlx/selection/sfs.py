from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ..utils import check_1d_array, check_2d_array, clone


class SequentialFeatureSelector(BaseTransformer):
    """Forward or backward sequential feature selection."""

    def __init__(self, estimator, n_features_to_select=None, direction="forward",
                 scoring=None, cv=5):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.direction = direction
        self.scoring = scoring
        self.cv = cv
        self.support_ = None
        self.scores_ = []

    def _score(self, X, y, mask):
        from ..data.model_selection import cross_val_score
        from ..data.cv import KFold
        est = clone(self.estimator)
        X_sub = X[:, mask]
        if X_sub.shape[1] == 0:
            return -np.inf
        cv = self.cv if hasattr(self.cv, 'split') else KFold(n_splits=int(self.cv), shuffle=True, random_state=0)
        scores = cross_val_score(est, X_sub, y, cv=cv, scoring=self.scoring)
        return float(np.mean(scores))

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = check_1d_array(y) if y is not None else None
        if y is None and self.scoring is not None:
            raise ValueError("y must be provided when scoring is not None")
        d = X.shape[1]
        n_select = self.n_features_to_select or max(1, d // 2)
        if n_select < 1 or n_select > d:
            raise ValueError(f"n_features_to_select must be in [1, {d}]")

        if self.direction == "forward":
            selected: list[int] = []
            remaining = list(range(d))
            self.scores_ = []
            for _ in range(n_select):
                best_score = -np.inf
                best_feat = -1
                for feat in remaining:
                    mask = np.array(sorted(selected + [feat]))
                    score = self._score(X, y, mask)
                    if score > best_score:
                        best_score = score
                        best_feat = feat
                if best_feat == -1:
                    break
                selected.append(best_feat)
                remaining.remove(best_feat)
                self.scores_.append(best_score)
        else:
            selected = list(range(d))
            self.scores_ = []
            for _ in range(d - n_select):
                best_score = -np.inf
                best_feat = -1
                for feat in selected:
                    mask = np.array(sorted([f for f in selected if f != feat]))
                    score = self._score(X, y, mask)
                    if score > best_score:
                        best_score = score
                        best_feat = feat
                if best_feat == -1:
                    break
                selected.remove(best_feat)
                self.scores_.append(best_score)

        mask = np.zeros(d, dtype=bool)
        mask[selected] = True
        self.support_ = mask
        return self

    def transform(self, X):
        X = check_2d_array(X)
        if self.support_ is None:
            raise RuntimeError("Selector has not been fit yet")
        return X[:, self.support_]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
