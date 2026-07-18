from __future__ import annotations

import numpy as np
from typing import cast

from ..base import BaseEstimator
from ..utils import accuracy, as_2d, check_1d_array, check_2d_array, r2_score


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
        self.estimators_ = []
        for est in self.estimators:
            est.fit(X, y)
            self.estimators_.append(est)
        self.classes_ = np.asarray(getattr(self.estimators_[0], "classes_", np.unique(y)))
        return self

    def _w(self):
        if self.weights is None:
            return np.ones(len(self.estimators_), dtype=float)
        if self.weights.shape[0] != len(self.estimators_):
            raise ValueError("weights must match number of estimators")
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
        return probs[0] if probs.shape[0] == 1 else probs

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
        out: list[object] = []
        for col in preds.T:
            vals, cnt = np.unique(col, return_counts=True)
            out.append(vals[np.argmax(cnt)])
        result = np.asarray(out)
        return result[0] if result.shape[0] == 1 else result

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
        self.estimators_ = []
        for est in self.estimators:
            est.fit(X, y)
            self.estimators_.append(est)
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
            w = self.weights / np.sum(self.weights)
            out = np.average(preds, axis=0, weights=w)
        return float(out[0]) if out.shape[0] == 1 else out

    def score(self, X, y):
        return r2_score(y, self.predict(X))
