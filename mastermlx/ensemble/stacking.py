from __future__ import annotations

import numpy as np
from typing import Any

from ..base import BaseEstimator
from ..data.cv import KFold
from ..linear_models import LinearRegression, LogisticRegression
from ..utils import accuracy, as_2d, check_1d_array, check_2d_array, clone, r2_score


def _oof_preds(estimators, X, y, cv, method):
    X = np.asarray(X)
    y = np.asarray(y)
    splits = list(cv.split(X, y))
    feats = None
    for train_idx, test_idx in splits:
        row = []
        for est in estimators:
            model = clone(est)
            model.fit(X[train_idx], y[train_idx])
            if method == "predict_proba" and not hasattr(model, "predict_proba"):
                pred = model.predict(X[test_idx])
            else:
                pred = getattr(model, method)(X[test_idx])
            pred = np.asarray(pred)
            if pred.ndim == 1:
                pred = pred[:, None]
            row.append(pred)
        block = np.concatenate(row, axis=1)
        if feats is None:
            feats = np.empty((X.shape[0], block.shape[1]), dtype=block.dtype)
        feats[test_idx] = block
    return feats


class StackingClassifier(BaseEstimator):
    def __init__(self, estimators, final_estimator=None, cv=5, use_proba=True, random_state=None):
        self.estimators = list(estimators)
        self.final_estimator = final_estimator or LogisticRegression()
        self.cv = cv
        self.use_proba = use_proba
        self.random_state = random_state
        self.estimators_ = []
        self.final_estimator_: Any | None = None

    def _require_final(self):
        if self.final_estimator_ is None:
            raise RuntimeError("Model has not been fit yet")
        return self.final_estimator_

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = check_1d_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        cv = self.cv if hasattr(self.cv, "split") else KFold(n_splits=int(self.cv), shuffle=True, random_state=self.random_state)
        method = "predict_proba" if self.use_proba else "predict"
        feats = _oof_preds(self.estimators, X, y, cv, method)
        self.estimators_ = []
        for est in self.estimators:
            est.fit(X, y)
            self.estimators_.append(est)
        self.final_estimator_ = clone(self.final_estimator).fit(feats, y)
        return self

    def _feat(self, X):
        if not self.estimators_:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X)
        parts = []
        for est in self.estimators_:
            pred = est.predict_proba(X) if self.use_proba and hasattr(est, "predict_proba") else est.predict(X)
            pred = np.asarray(pred)
            if pred.ndim == 1:
                pred = pred[:, None]
            parts.append(pred)
        return np.concatenate(parts, axis=1)

    def predict(self, X):
        feats = self._feat(X)
        pred = self._require_final().predict(feats)
        return pred[0] if np.asarray(pred).shape[0] == 1 else pred

    def predict_proba(self, X):
        feats = self._feat(X)
        final = self._require_final()
        if not hasattr(final, "predict_proba"):
            raise AttributeError("final_estimator does not define predict_proba")
        proba = final.predict_proba(feats)
        return proba[0] if proba.shape[0] == 1 else proba

    def score(self, X, y):
        return accuracy(y, self.predict(X))


class StackingRegressor(BaseEstimator):
    def __init__(self, estimators, final_estimator=None, cv=5, use_pred=True, random_state=None):
        self.estimators = list(estimators)
        self.final_estimator = final_estimator or LinearRegression()
        self.cv = cv
        self.random_state = random_state
        self.use_pred = use_pred
        self.estimators_ = []
        self.final_estimator_: Any | None = None

    def _require_final(self):
        if self.final_estimator_ is None:
            raise RuntimeError("Model has not been fit yet")
        return self.final_estimator_

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = check_1d_array(y).astype(float)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        cv = self.cv if hasattr(self.cv, "split") else KFold(n_splits=int(self.cv), shuffle=True, random_state=self.random_state)
        feats = _oof_preds(self.estimators, X, y, cv, "predict")
        self.estimators_ = []
        for est in self.estimators:
            est.fit(X, y)
            self.estimators_.append(est)
        self.final_estimator_ = clone(self.final_estimator).fit(feats, y)
        return self

    def _feat(self, X):
        if not self.estimators_:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X)
        parts = [np.asarray(est.predict(X), dtype=float)[:, None] for est in self.estimators_]
        return np.concatenate(parts, axis=1)

    def predict(self, X):
        feats = self._feat(X)
        pred = self._require_final().predict(feats)
        return float(pred[0]) if np.asarray(pred).shape[0] == 1 else pred

    def score(self, X, y):
        return r2_score(y, self.predict(X))
