from __future__ import annotations

import numpy as np
from typing import cast

from ..base import BaseEstimator
from ..utils import accuracy, as_2d, check_1d_array, check_2d_array, clone, r2_score


class VotingClassifier(BaseEstimator):
    def __init__(self, estimators, weights=None, voting="hard"):
        self.estimators = list(estimators)
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        self.voting = voting
        self.estimators_ = []
        self.classes_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = check_1d_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if not self.estimators:
            raise ValueError("estimators must contain at least one estimator")
        self.estimators_ = []
        for est in self.estimators:
            self.estimators_.append(clone(est).fit(X, y))
        self.classes_ = np.asarray(getattr(self.estimators_[0], "classes_", np.unique(y)))
        return self

    def _w(self):
        if self.weights is None:
            return np.ones(len(self.estimators_), dtype=float)
        if self.weights.shape[0] != len(self.estimators_):
            raise ValueError("weights must match number of estimators")
        if not np.all(np.isfinite(self.weights)) or np.any(self.weights < 0.0):
            raise ValueError("weights must contain finite non-negative values")
        if np.sum(self.weights) <= 0.0:
            raise ValueError("weights must contain at least one positive value")
        return self.weights

    def predict_proba(self, X):
        if not self.estimators_:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X)
        ws = self._w()
        probs = None
        for w, est in zip(ws, self.estimators_):
            p = est.predict_proba(X)
            p = np.asarray(p, dtype=float)
            if probs is None:
                probs = np.zeros_like(p, dtype=float)
            probs += w * p
        probs /= np.sum(ws)
        return probs

    def predict(self, X):
        if self.voting == "soft":
            proba = self.predict_proba(X)
            if proba.ndim == 1:
                return cast(np.ndarray, self.classes_)[int(np.argmax(proba))]
            return cast(np.ndarray, self.classes_)[np.argmax(proba, axis=1)]
        if not self.estimators_:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X)
        preds = np.asarray([est.predict(X) for est in self.estimators_])
        weights = self._w()
        out: list[object] = []
        for col in preds.T:
            values = np.unique(col)
            scores = np.asarray([np.sum(weights[col == value]) for value in values])
            out.append(values[np.argmax(scores)])
        result = np.asarray(out)
        return result

    def score(self, X, y):
        return accuracy(y, self.predict(X))


class VotingRegressor(BaseEstimator):
    def __init__(self, estimators, weights=None):
        self.estimators = list(estimators)
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        self.estimators_ = []

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = check_1d_array(y).astype(float)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if not self.estimators:
            raise ValueError("estimators must contain at least one estimator")
        self.estimators_ = []
        for est in self.estimators:
            self.estimators_.append(clone(est).fit(X, y))
        return self

    def predict(self, X):
        if not self.estimators_:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X)
        preds = np.asarray([est.predict(X) for est in self.estimators_], dtype=float)
        if self.weights is None:
            out = np.mean(preds, axis=0)
        else:
            if self.weights.shape[0] != len(self.estimators_):
                raise ValueError("weights must match number of estimators")
            if not np.all(np.isfinite(self.weights)) or np.any(self.weights < 0.0):
                raise ValueError("weights must contain finite non-negative values")
            if np.sum(self.weights) <= 0.0:
                raise ValueError("weights must contain at least one positive value")
            w = self.weights / np.sum(self.weights)
            out = np.average(preds, axis=0, weights=w)
        return out

    def score(self, X, y):
        return r2_score(y, self.predict(X))
