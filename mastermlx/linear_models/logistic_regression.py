from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..base import BaseEstimator
from ..utils.math import sigmoid
from ..utils.validation import check_X, check_sample_weight, check_same_rows, check_y, to_dense


class LogisticRegression(BaseEstimator):
    """Binary or multiclass logistic regression trained with gradient descent."""

    def __init__(
        self,
        lr=0.1,
        n_iter=1000,
        batch_size=None,
        fit_intercept=True,
        tol=1e-6,
        random_state=None,
        warm_start=False,
    ):
        self.lr = lr
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.random_state = random_state
        self.warm_start = bool(warm_start)
        self.coef_ = None
        self.intercept_ = None
        self.loss_ = []
        self.n_iter_ = 0
        self.multi_class_ = False

    def _add_bias(self, X):
        if self.fit_intercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
    ) -> "LogisticRegression":
        X = np.ascontiguousarray(to_dense(check_X(X, dtype=float, ensure_all_finite=True)))
        y = check_y(y)
        X, y = check_same_rows(X, y)
        self._set_n_features(X)
        sample_weight = check_sample_weight(sample_weight, X.shape[0])
        if not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be positive and finite")
        n_iter = int(self.n_iter)
        if n_iter < 1:
            raise ValueError("n_iter must be at least 1")
        if self.batch_size is not None:
            batch_size = int(self.batch_size)
            if batch_size < 1:
                raise ValueError("batch_size must be at least 1")
        else:
            batch_size = X.shape[0]
        if self.tol < 0 or not np.isfinite(self.tol):
            raise ValueError("tol must be non-negative and finite")

        bs = min(batch_size, X.shape[0])
        full_batch = bs == X.shape[0]
        classes = np.unique(y)
        previous_classes = getattr(self, "classes_", None)
        previous_multi_class = self.multi_class_
        self.loss_ = []

        if classes.shape[0] == 2:
            self.multi_class_ = False
            y_min, y_max = classes[0], classes[1]
            y_bin = (y == y_max).astype(float)
            Xb = np.ascontiguousarray(self._add_bias(X), dtype=float)
            rng = np.random.default_rng(self.random_state)
            reuse = (
                self.warm_start
                and previous_classes is not None
                and np.array_equal(previous_classes, classes)
                and not previous_multi_class
                and self.coef_ is not None
                and np.asarray(self.coef_).shape == (X.shape[1],)
            )
            if reuse:
                w = (
                    np.concatenate([[float(self.intercept_)], np.asarray(self.coef_)])
                    if self.fit_intercept
                    else np.asarray(self.coef_).copy()
                )
            else:
                w = rng.normal(scale=0.01, size=Xb.shape[1])

            prev = None
            total_weight = float(np.sum(sample_weight))
            for epoch in range(n_iter):
                if full_batch:
                    z = Xb @ w
                    p = sigmoid(z)
                    w -= self.lr * (Xb.T @ (sample_weight * (p - y_bin))) / total_weight
                else:
                    indices = rng.permutation(X.shape[0])
                    for start in range(0, X.shape[0], bs):
                        batch_idx = indices[start:start + bs]
                        xb = Xb[batch_idx]
                        yb = y_bin[batch_idx]
                        weights = sample_weight[batch_idx]
                        weight_sum = float(np.sum(weights))
                        if weight_sum == 0.0:
                            continue
                        z = xb @ w
                        p = sigmoid(z)
                        w -= self.lr * (xb.T @ (weights * (p - yb))) / weight_sum

                z = Xb @ w
                loss = float(
                    np.sum(sample_weight * (np.logaddexp(0.0, z) - y_bin * z))
                    / total_weight
                )
                self.loss_.append(loss)
                self.n_iter_ = epoch + 1
                if prev is not None and abs(prev - loss) < self.tol:
                    break
                prev = loss

            if self.fit_intercept:
                self.intercept_ = float(w[0])
                self.coef_ = w[1:]
            else:
                self.intercept_ = 0.0
                self.coef_ = w
            self.classes_ = np.array([y_min, y_max])
            return self

        self.multi_class_ = True
        n_classes = classes.shape[0]
        y_idx = np.searchsorted(classes, y)
        y_onehot = np.eye(n_classes)[y_idx]
        Xb = np.ascontiguousarray(self._add_bias(X), dtype=float)
        rng = np.random.default_rng(self.random_state)
        reuse = (
            self.warm_start
            and previous_classes is not None
            and np.array_equal(previous_classes, classes)
            and previous_multi_class
            and self.coef_ is not None
            and np.asarray(self.coef_).shape == (X.shape[1], n_classes)
        )
        if reuse:
            W = (
                np.vstack([np.asarray(self.intercept_), np.asarray(self.coef_)])
                if self.fit_intercept
                else np.asarray(self.coef_).copy()
            )
        else:
            W = rng.normal(scale=0.01, size=(Xb.shape[1], n_classes))

        def _softmax(z):
            z = z - np.max(z, axis=1, keepdims=True)
            exp = np.exp(z)
            return exp / np.sum(exp, axis=1, keepdims=True)

        prev = None
        total_weight = float(np.sum(sample_weight))
        for epoch in range(n_iter):
            if full_batch:
                logits = Xb @ W
                p = _softmax(logits)
                W -= self.lr * (Xb.T @ (sample_weight[:, None] * (p - y_onehot))) / total_weight
            else:
                indices = rng.permutation(X.shape[0])
                for start in range(0, X.shape[0], bs):
                    batch_idx = indices[start:start + bs]
                    xb = Xb[batch_idx]
                    yb = y_onehot[batch_idx]
                    weights = sample_weight[batch_idx]
                    weight_sum = float(np.sum(weights))
                    if weight_sum == 0.0:
                        continue
                    logits = xb @ W
                    p = _softmax(logits)
                    W -= self.lr * (xb.T @ (weights[:, None] * (p - yb))) / weight_sum

            logits = Xb @ W
            shifted = logits - np.max(logits, axis=1, keepdims=True)
            log_norm = np.max(logits, axis=1) + np.log(np.sum(np.exp(shifted), axis=1))
            loss = float(
                np.sum(sample_weight * (log_norm - np.sum(y_onehot * logits, axis=1)))
                / total_weight
            )
            self.loss_.append(loss)
            self.n_iter_ = epoch + 1
            if prev is not None and abs(prev - loss) < self.tol:
                break
            prev = loss

        if self.fit_intercept:
            self.intercept_ = W[0]
            self.coef_ = W[1:]
        else:
            self.intercept_ = np.zeros(n_classes)
            self.coef_ = W
        self.classes_ = classes
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        X = to_dense(self._check_X(X, dtype=float, ensure_all_finite=True))
        if self.coef_ is None:
            raise RuntimeError("Model has not been fit yet")
        if self.multi_class_:
            z = X @ self.coef_ + self.intercept_
            z = z - np.max(z, axis=1, keepdims=True)
            p = np.exp(z)
            return p / np.sum(p, axis=1, keepdims=True)
        z = X @ self.coef_ + self.intercept_
        p1 = sigmoid(z)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: ArrayLike) -> np.ndarray:
        proba = self.predict_proba(X)
        if self.multi_class_:
            return self.classes_[np.argmax(proba, axis=1)]
        return self.classes_[(proba[:, 1] >= 0.5).astype(int)]

    def score(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> float:
        y = check_y(y)
        pred = self.predict(X)
        if y.shape != pred.shape:
            raise ValueError("y and predictions must have the same shape")
        weights = check_sample_weight(sample_weight, y.shape[0])
        return float(np.average(y == pred, weights=weights))
