from __future__ import annotations

import numpy as np

from ...base import BaseLayer


class GlobalAveragePooling1D(BaseLayer):
    """Average over a sequence axis while keeping batch dimension."""

    def __init__(self):
        self.seq_len_ = None

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.size == 0:
            raise ValueError("Expected a non-empty array")
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {X.shape}")
        self.seq_len_ = X.shape[1]
        return np.mean(X, axis=1)

    def backward(self, grad):
        grad = np.asarray(grad, dtype=float)
        if self.seq_len_ is None:
            raise RuntimeError("forward must be called before backward")
        if grad.ndim != 2:
            raise ValueError(f"Expected 2D gradient, got shape {grad.shape}")
        return np.repeat(grad[:, None, :], self.seq_len_, axis=1) / self.seq_len_
