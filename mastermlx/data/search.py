"""Cross-validated grid and randomized search estimators."""

from __future__ import annotations

import numpy as np
from typing import Any

from ..base import BaseEstimator
from .search_core import (
    _param_grid_iter,
    _sample_param_distributions,
    _summarize_search,
    sample_params,
)


class GridSearchCV(BaseEstimator):
    """Grid search over a parameter grid with cross-validation."""

    def __init__(self, estimator, param_grid, cv=None, scoring=None, refit=True, return_train_score=False, error_score=np.nan):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.return_train_score = return_train_score
        self.error_score = error_score
        self.best_estimator_: Any | None = None
        self.best_params_ = None
        self.best_score_ = None
        self.best_index_ = None
        self.cv_results_ = None

    def fit(self, X, y=None, groups=None):
        X = np.asarray(X)
        y = np.asarray(y)
        candidates = list(_param_grid_iter(self.param_grid))
        result = _summarize_search(
            self.estimator,
            candidates,
            X,
            y,
            cv=self.cv,
            scoring=self.scoring,
            refit=self.refit,
            return_train_score=self.return_train_score,
            groups=groups,
            error_score=self.error_score,
        )
        self.best_estimator_, self.best_params_, self.best_score_, self.best_index_, self.cv_results_ = result
        return self

    def _require_best(self):
        if self.best_estimator_ is None:
            raise RuntimeError("GridSearchCV has not been fit with refit=True")
        return self.best_estimator_

    def predict(self, X):
        best = self._require_best()
        return best.predict(X)

    def predict_proba(self, X):
        best = self._require_best()
        if not hasattr(best, "predict_proba"):
            raise AttributeError("Best estimator does not define predict_proba")
        return best.predict_proba(X)

    def decision_function(self, X):
        best = self._require_best()
        if not hasattr(best, "decision_function"):
            raise AttributeError("Best estimator does not define decision_function")
        return best.decision_function(X)

    def score(self, X, y):
        best = self._require_best()
        return best.score(X, y)


class RandomizedSearchCV(BaseEstimator):
    """Randomized parameter search with cross-validation."""

    def __init__(
        self,
        estimator,
        param_distributions,
        n_iter=10,
        cv=None,
        scoring=None,
        refit=True,
        return_train_score=False,
        random_state=None,
        error_score=np.nan,
    ):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = int(n_iter)
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.return_train_score = return_train_score
        self.random_state = random_state
        self.error_score = error_score
        self.best_estimator_: Any | None = None
        self.best_params_ = None
        self.best_score_ = None
        self.best_index_ = None
        self.cv_results_ = None

    def fit(self, X, y=None, groups=None):
        X = np.asarray(X)
        y = np.asarray(y)
        if self.n_iter < 1:
            raise ValueError("n_iter must be at least 1")
        candidates = _sample_param_distributions(
            self.param_distributions,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        result = _summarize_search(
            self.estimator,
            candidates,
            X,
            y,
            cv=self.cv,
            scoring=self.scoring,
            refit=self.refit,
            return_train_score=self.return_train_score,
            groups=groups,
            error_score=self.error_score,
        )
        self.best_estimator_, self.best_params_, self.best_score_, self.best_index_, self.cv_results_ = result
        return self

    def _require_best(self):
        if self.best_estimator_ is None:
            raise RuntimeError("RandomizedSearchCV has not been fit with refit=True")
        return self.best_estimator_

    def predict(self, X):
        best = self._require_best()
        return best.predict(X)

    def predict_proba(self, X):
        best = self._require_best()
        if not hasattr(best, "predict_proba"):
            raise AttributeError("Best estimator does not define predict_proba")
        return best.predict_proba(X)

    def decision_function(self, X):
        best = self._require_best()
        if not hasattr(best, "decision_function"):
            raise AttributeError("Best estimator does not define decision_function")
        return best.decision_function(X)

    def score(self, X, y):
        best = self._require_best()
        return best.score(X, y)


__all__ = ["GridSearchCV", "RandomizedSearchCV", "sample_params"]
