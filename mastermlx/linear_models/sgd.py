from __future__ import annotations

import numpy as np
from typing import cast

from ..base import BaseEstimator
from ..utils.array import batch_iterator
from ..utils.validation import check_2d_array, check_1d_array, check_same_rows


# ---------------------------------------------------------------------------
# Loss functions and their gradients
# ---------------------------------------------------------------------------

def _hinge_loss(decision, y):
    """Hinge loss for binary labels in {0,1} or {-1,1}."""
    y = np.asarray(y, dtype=float)
    if np.any(y == 0):
        y = 2.0 * y - 1.0  # {0,1} -> {-1,1}
    margins = 1.0 - y * decision
    return float(np.mean(np.maximum(0.0, margins)))


def _hinge_grad(decision, y):
    y = np.asarray(y, dtype=float)
    if np.any(y == 0):
        y = 2.0 * y - 1.0
    mask = y * decision < 1.0
    grad = np.zeros_like(decision, dtype=float)
    grad[mask] = -y[mask]
    return grad


def _log_loss(decision, y):
    """Logistic loss for {-1,1} labels."""
    y = np.asarray(y, dtype=float)
    # numerically stable: log(1 + exp(-y*d))
    z = -y * decision
    # clip to avoid overflow
    loss = np.where(z > 50, z, np.log1p(np.exp(np.clip(z, -50, 50))))
    return float(np.mean(loss))


def _log_grad(decision, y):
    y = np.asarray(y, dtype=float)
    z = -y * decision
    # gradient: -y / (1 + exp(y*d))
    exp_z = np.exp(np.clip(z, -50, 50))
    return -y * exp_z / (1.0 + exp_z)


def _squared_loss(pred, y):
    return float(np.mean((y - pred) ** 2))


def _squared_grad(pred, y):
    return 2.0 * (pred - y)


def _huber_loss(pred, y, delta=1.0):
    diff = pred - y
    abs_diff = np.abs(diff)
    quad = 0.5 * diff ** 2
    linear = delta * (abs_diff - 0.5 * delta)
    return float(np.mean(np.where(abs_diff <= delta, quad, linear)))


def _huber_grad(pred, y, delta=1.0):
    diff = pred - y
    return np.where(np.abs(diff) <= delta, diff, np.sign(diff) * delta)


def _eps_insensitive_loss(pred, y, eps=0.1):
    return float(np.mean(np.maximum(0.0, np.abs(pred - y) - eps)))


def _eps_insensitive_grad(pred, y, eps=0.1):
    diff = pred - y
    grad = np.zeros_like(diff, dtype=float)
    mask_pos = diff > eps
    mask_neg = diff < -eps
    grad[mask_pos] = 1.0
    grad[mask_neg] = -1.0
    return grad


# ---------------------------------------------------------------------------
# Base SGD estimator
# ---------------------------------------------------------------------------

class _BaseSGD(BaseEstimator):
    """Common SGD logic for linear models."""

    def __init__(self, loss, penalty="l2", alpha=0.0001, l1_ratio=0.15,
                 max_iter=1000, tol=1e-3, learning_rate="constant",
                 eta0=0.01, batch_size=None, shuffle=True, random_state=None,
                 warm_start=False, average=False):
        self.loss = loss
        self.penalty = penalty
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.learning_rate = learning_rate
        self.eta0 = float(eta0)
        self.batch_size = batch_size
        self.shuffle = bool(shuffle)
        self.random_state = random_state
        self.warm_start = bool(warm_start)
        self.average = bool(average)
        self.coef_ = None
        self.intercept_ = None
        self.loss_curve_ = []

    def _init_weights(self, n_features):
        rng = np.random.default_rng(self.random_state)
        if not self.warm_start or self.coef_ is None:
            self.coef_ = rng.normal(scale=0.01, size=n_features)
            self.intercept_ = 0.0

    def _decision(self, X):
        return X @ cast(np.ndarray, self.coef_) + cast(float, self.intercept_)

    def _apply_l1_l2(self, lr, alpha_scaled):
        """Apply L2 shrinkage and L1 soft-threshold (proximal gradient)."""
        if self.penalty in {"l2", "elasticnet"}:
            self.coef_ = cast(np.ndarray, self.coef_) * (1.0 - lr * alpha_scaled * (1.0 - self.l1_ratio))
        if self.penalty in {"l1", "elasticnet"}:
            threshold = lr * alpha_scaled * self.l1_ratio
            coef = cast(np.ndarray, self.coef_)
            self.coef_ = np.sign(coef) * np.maximum(np.abs(coef) - threshold, 0.0)

    def _lrate(self, t):
        t = max(t, 1)
        if self.learning_rate == "constant":
            return self.eta0
        if self.learning_rate == "optimal":
            return 1.0 / (self.alpha * t)
        if self.learning_rate == "invscaling":
            return self.eta0 / np.power(t, 0.5)
        return self.eta0

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        y = check_1d_array(y).astype(float)
        X, y = check_same_rows(X, y)
        n, d = X.shape

        self._init_weights(d)
        bs = min(self.batch_size or n, n)
        rng = np.random.default_rng(self.random_state)
        self.loss_curve_ = []
        prev_loss = np.inf
        coef = cast(np.ndarray, self.coef_)
        avg_coef = np.zeros_like(coef)
        avg_intercept = 0.0
        t = 0

        for epoch in range(self.max_iter):
            for xb, yb in batch_iterator(X, y, batch_size=bs, shuffle=self.shuffle,
                                          random_state=rng.integers(0, 1 << 31)):
                t += 1
                lr = self._lrate(t)
                decision = xb @ cast(np.ndarray, self.coef_) + cast(float, self.intercept_)
                grad_loss = self._loss_grad(decision, yb)
                self.coef_ -= lr * (xb.T @ grad_loss / xb.shape[0])
                self._apply_l1_l2(lr, self.alpha)
                self.intercept_ = cast(float, self.intercept_) - lr * np.mean(grad_loss)

                if self.average:
                    avg_coef = avg_coef + (cast(np.ndarray, self.coef_) - avg_coef) / max(t, 1)
                    avg_intercept += (cast(float, self.intercept_) - avg_intercept) / max(t, 1)

            full_decision = X @ cast(np.ndarray, self.coef_) + cast(float, self.intercept_)
            loss = self._loss(full_decision, y) + self._reg_penalty()
            self.loss_curve_.append(loss)

            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        if self.average and t > 0:
            self.coef_ = avg_coef
            self.intercept_ = avg_intercept
        return self

    def _reg_penalty(self):
        if self.penalty not in {"l1", "l2", "elasticnet"} or self.alpha == 0:
            return 0.0
        penalty = 0.0
        if self.penalty in {"l2", "elasticnet"}:
            penalty += 0.5 * (1.0 - self.l1_ratio) * np.sum(cast(np.ndarray, self.coef_) ** 2)
        if self.penalty in {"l1", "elasticnet"}:
            penalty += self.l1_ratio * np.sum(np.abs(cast(np.ndarray, self.coef_)))
        return self.alpha * penalty

    def _loss(self, decision, y):
        raise NotImplementedError

    def _loss_grad(self, decision, y):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# SGDClassifier
# ---------------------------------------------------------------------------

class SGDClassifier(_BaseSGD):
    """Linear classifier trained with SGD."""

    def __init__(self, loss="hinge", penalty="l2", alpha=0.0001, l1_ratio=0.15,
                 max_iter=1000, tol=1e-3, learning_rate="constant",
                 eta0=0.01, batch_size=None, shuffle=True, random_state=None,
                 warm_start=False, average=False):
        if loss not in {"hinge", "log_loss", "modified_huber"}:
            raise ValueError("loss must be one of: hinge, log_loss, modified_huber")
        super().__init__(loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
                         max_iter=max_iter, tol=tol, learning_rate=learning_rate,
                         eta0=eta0, batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, warm_start=warm_start, average=average)
        self.classes_ = None

    def fit(self, X, y=None):
        y = check_1d_array(y).astype(float)
        X = check_2d_array(X).astype(float)
        X, y = check_same_rows(X, y)
        self.classes_ = np.unique(y)
        classes = cast(np.ndarray, self.classes_)
        if classes.size < 2:
            raise ValueError("Need at least two classes")
        if classes.size == 2:
            # Binary: map to +1 / -1
            y_bin = np.where(y == classes[1], 1.0, -1.0)
        else:
            # Multiclass: one-vs-rest stored as list
            self._coefs_: list[np.ndarray] = []
            self._intercepts_: list[float] = []
            for c in classes:
                y_bin_c = np.where(y == c, 1.0, -1.0)
                est = _BinarySGDClassifier(
                    loss=self.loss, penalty=self.penalty, alpha=self.alpha,
                    l1_ratio=self.l1_ratio, max_iter=self.max_iter, tol=self.tol,
                    learning_rate=self.learning_rate, eta0=self.eta0,
                    batch_size=self.batch_size, shuffle=self.shuffle,
                    random_state=self.random_state, warm_start=self.warm_start,
                    average=self.average,
                )
                est.fit(X, y_bin_c)
                self._coefs_.append(cast(np.ndarray, est.coef_))
                self._intercepts_.append(cast(float, est.intercept_))
            self.coef_ = np.column_stack(self._coefs_)
            self.intercept_ = np.array(self._intercepts_)
            return self

        super().fit(X, y_bin)
        return self

    def decision_function(self, X):
        X = check_2d_array(X).astype(float)
        if hasattr(self, '_coefs_'):
            return np.column_stack([X @ c + i for c, i in zip(self._coefs_, self._intercepts_)])
        return X @ self.coef_ + self.intercept_

    def predict(self, X):
        scores = self.decision_function(X)
        classes = cast(np.ndarray, self.classes_)
        if scores.ndim == 1:
            return np.where(scores >= 0, classes[1], classes[0])
        return classes[np.argmax(scores, axis=1)]

    def _loss(self, decision, y):
        if self.loss == "hinge":
            return _hinge_loss(decision, y)
        if self.loss == "log_loss":
            return _log_loss(decision, y)
        return _hinge_loss(decision, y)

    def _loss_grad(self, decision, y):
        if self.loss == "hinge":
            return _hinge_grad(decision, y)
        if self.loss == "log_loss":
            return _log_grad(decision, y)
        return _hinge_grad(decision, y)

    def score(self, X, y):
        from ..utils.metrics import accuracy
        return accuracy(y, self.predict(X))


class _BinarySGDClassifier(_BaseSGD):
    """Internal binary SGD classifier used by one-vs-rest."""

    def _loss(self, decision, y):
        if self.loss == "hinge":
            return _hinge_loss(decision, y)
        if self.loss == "log_loss":
            return _log_loss(decision, y)
        return _hinge_loss(decision, y)

    def _loss_grad(self, decision, y):
        if self.loss == "hinge":
            return _hinge_grad(decision, y)
        if self.loss == "log_loss":
            return _log_grad(decision, y)
        return _hinge_grad(decision, y)


# ---------------------------------------------------------------------------
# SGDRegressor
# ---------------------------------------------------------------------------

class SGDRegressor(_BaseSGD):
    """Linear regressor trained with SGD."""

    def __init__(self, loss="squared_error", penalty="l2", alpha=0.0001, l1_ratio=0.15,
                 max_iter=1000, tol=1e-3, learning_rate="constant",
                 eta0=0.01, batch_size=None, shuffle=True, random_state=None,
                 warm_start=False, average=False, delta=1.0, epsilon=0.1):
        if loss not in {"squared_error", "huber", "epsilon_insensitive"}:
            raise ValueError("loss must be one of: squared_error, huber, epsilon_insensitive")
        super().__init__(loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
                         max_iter=max_iter, tol=tol, learning_rate=learning_rate,
                         eta0=eta0, batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, warm_start=warm_start, average=average)
        self.delta = float(delta)
        self.epsilon = float(epsilon)

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        y = check_1d_array(y).astype(float)
        X, y = check_same_rows(X, y)
        return super().fit(X, y)

    def predict(self, X):
        X = check_2d_array(X).astype(float)
        if self.coef_ is None:
            raise RuntimeError("Model has not been fit yet")
        return self._decision(X)

    def _loss(self, decision, y):
        if self.loss == "squared_error":
            return _squared_loss(decision, y)
        if self.loss == "huber":
            return _huber_loss(decision, y, self.delta)
        return _eps_insensitive_loss(decision, y, self.epsilon)

    def _loss_grad(self, decision, y):
        if self.loss == "squared_error":
            return _squared_grad(decision, y)
        if self.loss == "huber":
            return _huber_grad(decision, y, self.delta)
        return _eps_insensitive_grad(decision, y, self.epsilon)

    def score(self, X, y):
        from ..utils.metrics import r2_score
        return r2_score(y, self.predict(X))
