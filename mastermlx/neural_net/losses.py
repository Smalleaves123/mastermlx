from __future__ import annotations

import numpy as np


class MSELoss:
    """Mean squared error loss."""

    def __call__(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.mean((y_true - y_pred) ** 2))

    def grad(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return 2.0 * (y_pred - y_true) / y_true.shape[0]


class CrossEntropyLoss:
    """Multiclass cross-entropy loss."""

    def __init__(self, from_logits=True, eps=1e-12):
        self.from_logits = from_logits
        self.eps = eps

    def _softmax(self, logits):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def __call__(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        probs = self._softmax(y_pred) if self.from_logits else y_pred
        return float(-np.mean(np.sum(y_true * np.log(probs + self.eps), axis=1)))

    def grad(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if self.from_logits:
            probs = self._softmax(y_pred)
            return (probs - y_true) / y_true.shape[0]
        return -(y_true / (y_pred + self.eps)) / y_true.shape[0]


class BinaryCrossEntropyLoss:
    """Binary cross-entropy for independent sigmoid outputs."""

    def __init__(self, from_logits=True, eps=1e-12):
        self.from_logits = bool(from_logits)
        self.eps = float(eps)

    def __call__(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if self.from_logits:
            value = np.maximum(y_pred, 0.0) - y_true * y_pred + np.log1p(np.exp(-np.abs(y_pred)))
            return float(np.mean(value))
        probs = np.clip(y_pred, self.eps, 1.0 - self.eps)
        return float(-np.mean(y_true * np.log(probs) + (1.0 - y_true) * np.log1p(-probs)))

    def grad(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if self.from_logits:
            positive = y_pred >= 0.0
            probs = np.empty_like(y_pred)
            probs[positive] = 1.0 / (1.0 + np.exp(-y_pred[positive]))
            exp_logits = np.exp(y_pred[~positive])
            probs[~positive] = exp_logits / (1.0 + exp_logits)
            return (probs - y_true) / y_true.shape[0]
        return (y_pred - y_true) / (np.maximum(y_pred * (1.0 - y_pred), self.eps) * y_true.shape[0])
