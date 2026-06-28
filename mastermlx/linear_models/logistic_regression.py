from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils.array import batch_iterator
from ..utils.math import sigmoid
from ..utils.metrics import accuracy
from ..utils.validation import check_1d_array, check_2d_array, check_same_rows


class LogisticRegression(BaseEstimator):
    """Binary or multiclass logistic regression trained with gradient descent."""

    def __init__(self, lr=0.1, n_iter=1000, batch_size=None, fit_intercept=True,
                 tol=1e-6, random_state=None):
        self.lr = lr
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None
        self.loss_ = []
        self.multi_class_ = False

    def _add_bias(self, X):
        if self.fit_intercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = check_1d_array(y)
        X, y = check_same_rows(X, y)
        bs = min(self.batch_size or X.shape[0], X.shape[0])

        classes = np.unique(y)
        self.loss_ = []

        if classes.shape[0] == 2:
            self.multi_class_ = False
            y_min, y_max = classes[0], classes[1]
            y_bin = (y == y_max).astype(float)

            Xb = self._add_bias(X)
            rng = np.random.default_rng(self.random_state)
            w = rng.normal(scale=0.01, size=Xb.shape[1])

            prev = None
            for epoch in range(self.n_iter):
                for xb, yb in batch_iterator(Xb, y_bin, batch_size=bs, shuffle=True,
                                              random_state=rng.integers(0, 1 << 31)):
                    z = xb @ w
                    p = sigmoid(z)
                    grad = (xb.T @ (p - yb)) / xb.shape[0]
                    w -= self.lr * grad

                z = Xb @ w
                p = sigmoid(z)
                eps = 1e-12
                loss = -np.mean(y_bin * np.log(p + eps) + (1.0 - y_bin) * np.log(1.0 - p + eps))
                self.loss_.append(loss)

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

        Xb = self._add_bias(X)
        rng = np.random.default_rng(self.random_state)
        W = rng.normal(scale=0.01, size=(Xb.shape[1], n_classes))

        def _softmax(z):
            z = z - np.max(z, axis=1, keepdims=True)
            exp = np.exp(z)
            return exp / np.sum(exp, axis=1, keepdims=True)

        prev = None
        for epoch in range(self.n_iter):
            for xb, yb in batch_iterator(Xb, y_onehot, batch_size=bs, shuffle=True,
                                          random_state=rng.integers(0, 1 << 31)):
                logits = xb @ W
                p = _softmax(logits)
                grad = (xb.T @ (p - yb)) / xb.shape[0]
                W -= self.lr * grad

            logits = Xb @ W
            p = _softmax(logits)
            eps = 1e-12
            loss = -np.mean(np.sum(y_onehot * np.log(p + eps), axis=1))
            self.loss_.append(loss)

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

    def predict_proba(self, X):
        X = check_2d_array(X)
        if self.coef_ is None:
            raise RuntimeError("Model has not been fit yet")
        if self.multi_class_:
            z = X @ self.coef_ + self.intercept_
            z = z - np.max(z, axis=1, keepdims=True)
            p = np.exp(z)
            p = p / np.sum(p, axis=1, keepdims=True)
            return p[0] if p.shape[0] == 1 else p
        z = X @ self.coef_ + self.intercept_
        p1 = sigmoid(z)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        proba = self.predict_proba(X)
        if self.multi_class_:
            idx = np.argmax(proba, axis=1)
            pred = self.classes_[idx]
            return pred[0] if pred.shape[0] == 1 else pred
        idx = (proba[:, 1] >= 0.5).astype(int)
        return self.classes_[idx]

    def score(self, X, y):
        return accuracy(y, self.predict(X))
