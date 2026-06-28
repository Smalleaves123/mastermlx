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
