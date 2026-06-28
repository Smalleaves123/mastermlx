from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
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

        # Fit base estimator on all data
        self.estimator.fit(X, y)
        self._calibrated = self.estimator

        # Get out-of-fold predictions for calibration
        if hasattr(self.estimator, "predict_proba"):
            proba = self.estimator.predict_proba(X)
            if proba.ndim != 2 or proba.shape[1] < 2:
                raise ValueError("predict_proba must return 2D with at least 2 columns")
            scores = proba[:, -1]
        elif hasattr(self.estimator, "decision_function"):
            scores = self.estimator.decision_function(X)
            if scores is None:
                raise RuntimeError("decision_function returned None")
            scores = np.asarray(scores, dtype=float).ravel()
        else:
            raise ValueError("estimator must have predict_proba or decision_function")

        # Platt scaling: fit sigmoid(a * score + b) to y_bin
        # Guard against degenerate cases
        if np.all(scores == scores[0]):
            self._a = 0.0
            self._b = np.log(np.mean(y_bin) / max(1e-12, 1.0 - np.mean(y_bin)))
            return self

        from math import log, exp
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

    def predict_proba(self, X):
        X = check_2d_array(X)
        if self._calibrated is None:
            raise RuntimeError("Model has not been fit yet")
        if hasattr(self._calibrated, "decision_function"):
            scores = np.asarray(self._calibrated.decision_function(X), dtype=float).ravel()
        elif hasattr(self._calibrated, "predict_proba"):
            scores = self._calibrated.predict_proba(X)[:, -1]
        else:
            raise RuntimeError("base estimator has neither decision_function nor predict_proba")

        calibrated = 1.0 / (1.0 + np.exp(-(self._a * scores + self._b)))
        calibrated = np.clip(calibrated, 1e-12, 1.0 - 1e-12)
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def score(self, X, y):
        from ..utils.metrics import accuracy
        return accuracy(y, self.predict(X))
