"""Inductive self-training for classifiers with probability estimates."""

from __future__ import annotations

import copy

import numpy as np

from ..base import BaseEstimator
from ..utils.metrics import accuracy
from ..utils.validation import check_1d_array, check_2d_array


class SelfTrainingClassifier(BaseEstimator):
    """Iteratively add confident pseudo-labels to a base classifier.

    ``y`` uses ``unlabeled_value`` (``-1`` by default) for unknown targets.
    The base estimator must implement ``fit`` and ``predict_proba``.  The
    fitted estimator is inductive, so ``predict`` also accepts new samples.
    """

    def __init__(
        self,
        base_estimator,
        *,
        threshold=0.75,
        criterion="threshold",
        k_best=1,
        max_iter=10,
        unlabeled_value=-1,
    ):
        if not hasattr(base_estimator, "fit") or not hasattr(base_estimator, "predict_proba"):
            raise TypeError("base_estimator must implement fit() and predict_proba()")
        self.base_estimator = base_estimator
        self.threshold = float(threshold)
        self.criterion = str(criterion).lower()
        self.k_best = int(k_best)
        self.max_iter = int(max_iter)
        self.unlabeled_value = unlabeled_value
        self.estimator_ = None
        self.classes_ = None
        self.transduction_ = None
        self.label_distributions_ = None
        self.labeled_iter_ = None
        self.n_iter_ = 0
        self.n_pseudo_labels_ = 0

    def _validate_parameters(self):
        if not 0.0 < self.threshold <= 1.0 or not np.isfinite(self.threshold):
            raise ValueError("threshold must be finite and in (0, 1]")
        if self.criterion not in {"threshold", "k_best"}:
            raise ValueError("criterion must be 'threshold' or 'k_best'")
        if self.k_best < 1:
            raise ValueError("k_best must be at least 1")
        if self.max_iter < 0:
            raise ValueError("max_iter must be non-negative")

    def _fit_estimator(self, X, y):
        estimator = copy.deepcopy(self.base_estimator)
        estimator.fit(X, y)
        if not hasattr(estimator, "classes_"):
            raise ValueError("base_estimator must expose classes_ after fit")
        return estimator

    def fit(self, X, y=None):
        """Fit the base classifier and iteratively absorb confident samples."""

        self._validate_parameters()
        X = check_2d_array(X).astype(float)
        if y is None:
            raise ValueError("y is required and must contain unlabeled_value entries")
        y = check_1d_array(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        labeled = y != self.unlabeled_value
        if np.count_nonzero(labeled) < 2:
            raise ValueError("y must contain at least two labeled samples")
        if np.unique(y[labeled]).size < 2:
            raise ValueError("labeled samples must contain at least two classes")

        working_labels = y.copy()
        labeled_iter = np.full(y.shape[0], -1, dtype=int)
        labeled_iter[labeled] = 0
        self.n_iter_ = 0
        for iteration in range(1, self.max_iter + 1):
            estimator = self._fit_estimator(X[labeled], working_labels[labeled])
            unlabeled_indices = np.flatnonzero(~labeled)
            if unlabeled_indices.size == 0:
                self.n_iter_ = iteration - 1
                break
            probabilities = np.asarray(estimator.predict_proba(X[unlabeled_indices]), dtype=float)
            if probabilities.ndim != 2 or probabilities.shape[0] != unlabeled_indices.size:
                raise ValueError("base_estimator.predict_proba() must return a 2D batch array")
            confidence = np.max(probabilities, axis=1)
            if self.criterion == "threshold":
                selected = np.flatnonzero(confidence >= self.threshold)
            else:
                count = min(self.k_best, unlabeled_indices.size)
                selected = np.argsort(-confidence, kind="stable")[:count]
                selected = selected[confidence[selected] >= self.threshold]
            if selected.size == 0:
                self.n_iter_ = iteration - 1
                break
            selected_indices = unlabeled_indices[selected]
            classes = np.asarray(estimator.classes_)
            working_labels[selected_indices] = classes[np.argmax(probabilities[selected], axis=1)]
            labeled[selected_indices] = True
            labeled_iter[selected_indices] = iteration
            self.n_iter_ = iteration
        else:
            self.n_iter_ = self.max_iter

        self.estimator_ = self._fit_estimator(X[labeled], working_labels[labeled])
        self.classes_ = np.asarray(self.estimator_.classes_).copy()
        self.transduction_ = working_labels.copy()
        self.labeled_iter_ = labeled_iter
        self.n_pseudo_labels_ = int(np.count_nonzero(labeled_iter > 0))
        self.label_distributions_ = np.asarray(self.estimator_.predict_proba(X), dtype=float)
        return self

    def _check_self_fitted(self):
        if self.estimator_ is None:
            raise RuntimeError("Model has not been fit yet")

    def predict_proba(self, X=None):
        """Predict class probabilities for fitted or new samples."""

        self._check_self_fitted()
        if X is None:
            return self.label_distributions_
        estimator = self.estimator_
        if estimator is None:
            raise RuntimeError("Model has not been fit yet")
        return np.asarray(estimator.predict_proba(check_2d_array(X).astype(float)))

    def predict(self, X=None):
        """Predict class labels for fitted or new samples."""

        self._check_self_fitted()
        if X is None:
            return self.transduction_
        probabilities = np.asarray(self.predict_proba(X))
        classes = self.classes_
        if classes is None:
            raise RuntimeError("Model has not been fit yet")
        return classes[np.argmax(probabilities, axis=1)]

    def fit_predict(self, X, y=None):
        return self.fit(X, y).transduction_

    def score(self, X, y):
        return accuracy(y, self.predict(X))


__all__ = ["SelfTrainingClassifier"]
