from __future__ import annotations

import numpy as np
from typing import cast

from ..base import BaseEstimator
from ..data.cv import StratifiedKFold
from ..utils import clone
from ..utils.validation import check_1d_array, check_2d_array, check_same_rows


class CalibratedClassifierCV(BaseEstimator):
    """Probability calibration via Platt scaling (sigmoid) with cross-validation."""

    def __init__(self, estimator=None, cv=5, method="sigmoid"):
        self.estimator = estimator
        self.cv = cv
        self.method = method
        self._calibrated = None
        self.classes_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = check_1d_array(y)
        X, y = check_same_rows(X, y)
        if self.estimator is None:
            raise ValueError("estimator must be provided")

        self.classes_ = np.unique(y)
        if self.classes_.size == 0:
            raise ValueError("y must have at least one unique class")
        if self.classes_.size != 2:
            raise ValueError("CalibratedClassifierCV only supports binary classification")
        if X.shape[0] < 3:
            raise ValueError("Need at least 3 samples for calibration CV")

        pos_class = self.classes_[1]
        y_bin = (y == pos_class).astype(float)

        if self.method != "sigmoid":
            raise ValueError("method must be 'sigmoid'")
        splitter = (
            StratifiedKFold(n_splits=int(self.cv), shuffle=True, random_state=0)
            if isinstance(self.cv, (int, np.integer))
            else self.cv
        )
        if not hasattr(splitter, "split"):
            raise TypeError("cv must be an integer or expose split(X, y)")

        scores = np.empty(X.shape[0], dtype=float)
        filled = np.zeros(X.shape[0], dtype=bool)
        for train_idx, valid_idx in splitter.split(X, y):
            fold_estimator = clone(self.estimator).fit(X[train_idx], y[train_idx])
            scores[valid_idx] = self._uncalibrated_scores(
                fold_estimator,
                X[valid_idx],
                pos_class,
            )
            filled[valid_idx] = True
        if not np.all(filled):
            raise ValueError("cv splits must provide a validation score for every sample")

        self._calibrated = clone(self.estimator).fit(X, y)
        self._set_n_features(X)

        # Platt scaling: fit sigmoid(a * score + b) to y_bin
        # Guard against degenerate cases
        if np.all(scores == scores[0]):
            self._a = 0.0
            self._b = np.log(np.mean(y_bin) / max(1e-12, 1.0 - np.mean(y_bin)))
            return self

        def _platt_objective(params):
            a, b = params
            f = a * scores + b
            # clip to avoid overflow
            f = np.clip(f, -50, 50)
            return -np.mean(y_bin * f - np.log1p(np.exp(f)))

        # Grid search for a, b
        best_loss = np.inf
        self._a, self._b = 1.0, 0.0
        for a in np.linspace(-3.0, 3.0, 31):
            for b in np.linspace(-3.0, 3.0, 31):
                loss = _platt_objective((a, b))
                if loss < best_loss:
                    best_loss = loss
                    self._a, self._b = a, b

        return self

    @staticmethod
    def _uncalibrated_scores(estimator, X, pos_class):
        if hasattr(estimator, "decision_function"):
            scores = np.asarray(estimator.decision_function(X), dtype=float)
            if scores.ndim == 2:
                classes = np.asarray(getattr(estimator, "classes_", []))
                matches = np.flatnonzero(classes == pos_class)
                if matches.size != 1:
                    raise ValueError("decision_function columns must match estimator classes")
                scores = scores[:, int(matches[0])]
            return scores.ravel()
        if hasattr(estimator, "predict_proba"):
            proba = np.asarray(estimator.predict_proba(X), dtype=float)
            classes = np.asarray(getattr(estimator, "classes_", []))
            matches = np.flatnonzero(classes == pos_class)
            if proba.ndim != 2 or matches.size != 1:
                raise ValueError("predict_proba columns must match estimator classes")
            return proba[:, int(matches[0])]
        raise ValueError("estimator must have predict_proba or decision_function")

    def predict_proba(self, X):
        X = self._check_X(X)
        if self._calibrated is None:
            raise RuntimeError("Model has not been fit yet")
        scores = self._uncalibrated_scores(
            self._calibrated,
            X,
            cast(np.ndarray, self.classes_)[1],
        )

        calibrated = 1.0 / (1.0 + np.exp(-(self._a * scores + self._b)))
        calibrated = np.clip(calibrated, 1e-12, 1.0 - 1e-12)
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return cast(np.ndarray, self.classes_)[idx]

    def score(self, X, y):
        from ..utils.metrics import accuracy
        return accuracy(y, self.predict(X))
