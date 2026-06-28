from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import accuracy, as_2d, check_1d_array, check_2d_array, check_same_rows
from .decision_tree import DecisionTreeRegressor


def _softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


class GradientBoostingClassifier(BaseEstimator):
    """Gradient boosting classifier with multinomial deviance."""

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None,
    ):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.classes_ = None
        self.init_scores_ = None
        self.estimators_ = []
        self.loss_ = []

    def _check_params(self):
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        y = check_1d_array(y)
        X, y = check_same_rows(X, y)
        self._check_params()

        classes = np.unique(y)
        if classes.shape[0] < 2:
            raise ValueError("GradientBoostingClassifier requires at least 2 classes")
        self.classes_ = classes
        n_samples, _ = X.shape
        n_classes = classes.shape[0]
        y_idx = np.searchsorted(classes, y)
        y_onehot = np.eye(n_classes)[y_idx]

        class_prior = np.clip(y_onehot.mean(axis=0), 1e-12, 1.0)
        scores = np.tile(np.log(class_prior), (n_samples, 1))
        self.init_ = np.log(class_prior)
        self.estimators_ = []
        self.loss_ = []

        for _ in range(self.n_estimators):
            proba = _softmax(scores)
            residual = y_onehot - proba
            stage = []
            for k in range(n_classes):
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                )
                tree.fit(X, residual[:, k])
                update = tree.predict(X)
                scores[:, k] += self.learning_rate * update
                stage.append(tree)
            self.estimators_.append(stage)

            eps = 1e-12
            loss = -np.mean(np.sum(y_onehot * np.log(_softmax(scores) + eps), axis=1))
            self.loss_.append(loss)

        return self

    def decision_function(self, X):
        if self.classes_ is None:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X).astype(float)
        n_samples = X.shape[0]
        scores = np.tile(self.init_, (n_samples, 1))
        for stage in self.estimators_:
            for k, tree in enumerate(stage):
                scores[:, k] += self.learning_rate * tree.predict(X)
        if scores.shape[1] == 2:
            margin = scores[:, 1] - scores[:, 0]
            return float(margin[0]) if margin.shape[0] == 1 else margin
        return scores[0] if scores.shape[0] == 1 else scores

    def staged_decision_function(self, X):
        if self.classes_ is None:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X).astype(float)
        n_samples = X.shape[0]
        scores = np.tile(self.init_, (n_samples, 1))
        for stage in self.estimators_:
            for k, tree in enumerate(stage):
                scores[:, k] += self.learning_rate * tree.predict(X)
            if scores.shape[1] == 2:
                margin = scores[:, 1] - scores[:, 0]
                yield float(margin[0]) if margin.shape[0] == 1 else margin.copy()
            else:
                yield scores[0].copy() if scores.shape[0] == 1 else scores.copy()

    def staged_predict_proba(self, X):
        for scores in self.staged_decision_function(X):
            scores = np.asarray(scores, dtype=float)
            if scores.ndim == 0:
                p1 = 1.0 / (1.0 + np.exp(-scores))
                yield np.array([1.0 - p1, p1])
            elif scores.ndim == 1 and self.classes_.shape[0] == 2:
                p1 = 1.0 / (1.0 + np.exp(-scores))
                yield np.column_stack([1.0 - p1, p1]) if p1.ndim > 0 else np.array([1.0 - p1, p1])
            else:
                shifted = scores - np.max(scores, axis=1, keepdims=True)
                exp = np.exp(shifted)
                yield exp / np.sum(exp, axis=1, keepdims=True)

    def staged_predict(self, X):
        for proba in self.staged_predict_proba(X):
            proba = np.asarray(proba, dtype=float)
            if proba.ndim == 1:
                yield self.classes_[int(np.argmax(proba))]
            else:
                yield self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        if self.classes_ is None:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X).astype(float)
        n_samples = X.shape[0]
        scores = np.tile(self.init_, (n_samples, 1))
        for stage in self.estimators_:
            for k, tree in enumerate(stage):
                scores[:, k] += self.learning_rate * tree.predict(X)
        proba = _softmax(scores)
        return proba[0] if proba.shape[0] == 1 else proba

    def predict(self, X):
        proba = self.predict_proba(X)
        if proba.ndim == 1:
            return self.classes_[int(np.argmax(proba))]
        idx = np.argmax(proba, axis=1)
        pred = self.classes_[idx]
        return pred[0] if pred.shape[0] == 1 else pred

    def score(self, X, y):
        return accuracy(y, self.predict(X))


GBC = GradientBoostingClassifier
