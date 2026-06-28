from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import accuracy, as_2d, check_1d_array, check_2d_array


class _WeightedDecisionStump:
    def __init__(self):
        self.feature_ = None
        self.threshold_ = None
        self.left_class_ = None
        self.right_class_ = None

    def fit(self, X, y, sample_weight, classes):
        n_samples, n_features = X.shape
        best_error = np.inf

        for feature in range(n_features):
            values = X[:, feature]
            unique_values = np.unique(values)
            if unique_values.size == 1:
                continue
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask
                if not np.any(left_mask) or not np.any(right_mask):
                    continue

                left_class = self._majority_class(y[left_mask], sample_weight[left_mask], classes)
                right_class = self._majority_class(y[right_mask], sample_weight[right_mask], classes)
                pred = np.where(left_mask, left_class, right_class)
                error = float(np.sum(sample_weight[pred != y]))

                if error < best_error:
                    best_error = error
                    self.feature_ = feature
                    self.threshold_ = float(threshold)
                    self.left_class_ = left_class
                    self.right_class_ = right_class

        if self.feature_ is None:
            majority = self._majority_class(y, sample_weight, classes)
            self.feature_ = 0
            self.threshold_ = float("inf")
            self.left_class_ = majority
            self.right_class_ = majority

        return self

    def _majority_class(self, y, sample_weight, classes):
        scores = np.zeros(classes.shape[0], dtype=float)
        for idx, cls in enumerate(classes):
            scores[idx] = np.sum(sample_weight[y == cls])
        return classes[int(np.argmax(scores))]

    def predict(self, X):
        X = as_2d(X)
        pred = np.where(X[:, self.feature_] <= self.threshold_, self.left_class_, self.right_class_)
        return pred[0] if pred.shape[0] == 1 else pred


class AdaBoostClassifier(BaseEstimator):
    """AdaBoost classifier with weighted decision stumps."""

    def __init__(self, n_estimators=50, learning_rate=1.0, random_state=None):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.random_state = random_state
        self.estimators_ = []
        self.estimator_weights_ = np.empty(0, dtype=float)
        self.classes_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        y = check_1d_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")

        n_samples = X.shape[0]
        self.classes_ = np.unique(y)
        n_classes = self.classes_.shape[0]
        if n_classes < 2:
            raise ValueError("AdaBoostClassifier requires at least 2 classes")

        sample_weight = np.full(n_samples, 1.0 / n_samples, dtype=float)
        self.estimators_ = []
        estimator_weights = []

        for _ in range(self.n_estimators):
            stump = _WeightedDecisionStump().fit(X, y, sample_weight, self.classes_)
            pred = stump.predict(X)
            incorrect = pred != y
            error = float(np.sum(sample_weight[incorrect]))
            error = np.clip(error, 1e-12, 1.0 - 1e-12)

            if error >= 1.0 - (1.0 / n_classes):
                continue

            alpha = self.learning_rate * (
                np.log((1.0 - error) / error) + np.log(max(n_classes - 1, 1))
            )
            sample_weight *= np.exp(alpha * incorrect.astype(float))
            sample_weight /= np.sum(sample_weight)

            self.estimators_.append(stump)
            estimator_weights.append(alpha)

            if error <= 1e-12:
                break

        if not self.estimators_:
            raise RuntimeError("AdaBoostClassifier could not fit a useful weak learner")

        self.estimator_weights_ = np.asarray(estimator_weights, dtype=float)
        return self

    def decision_function(self, X):
        if not self.estimators_:
            raise RuntimeError("Model has not been fit yet")

        X = as_2d(X).astype(float)
        scores = np.zeros((X.shape[0], self.classes_.shape[0]), dtype=float)
        for alpha, estimator in zip(self.estimator_weights_, self.estimators_):
            pred = estimator.predict(X)
            for idx, cls in enumerate(self.classes_):
                scores[:, idx] += alpha * (pred == cls)
        return scores

    def staged_decision_function(self, X):
        if not self.estimators_:
            raise RuntimeError("Model has not been fit yet")

        X = as_2d(X).astype(float)
        scores = np.zeros((X.shape[0], self.classes_.shape[0]), dtype=float)
        for alpha, estimator in zip(self.estimator_weights_, self.estimators_):
            pred = estimator.predict(X)
            for idx, cls in enumerate(self.classes_):
                scores[:, idx] += alpha * (pred == cls)
            yield scores[0].copy() if scores.shape[0] == 1 else scores.copy()

    def staged_predict_proba(self, X):
        for scores in self.staged_decision_function(X):
            scores = np.asarray(scores, dtype=float)
            if scores.ndim == 1:
                shifted = scores - np.max(scores)
                exp = np.exp(shifted)
                yield exp / np.sum(exp)
            else:
                shifted = scores - np.max(scores, axis=1, keepdims=True)
                exp = np.exp(shifted)
                yield exp / np.sum(exp, axis=1, keepdims=True)

    def staged_predict(self, X):
        for scores in self.staged_decision_function(X):
            scores = np.asarray(scores, dtype=float)
            if scores.ndim == 1:
                yield self.classes_[int(np.argmax(scores))]
            else:
                yield self.classes_[np.argmax(scores, axis=1)]

    def predict(self, X):
        scores = self.decision_function(X)
        pred = self.classes_[np.argmax(scores, axis=1)]
        return pred[0] if pred.shape[0] == 1 else pred

    def score(self, X, y):
        return accuracy(y, self.predict(X))


AdaBoostClf = AdaBoostClassifier
