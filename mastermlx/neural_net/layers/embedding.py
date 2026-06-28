from __future__ import annotations

import numpy as np

from ...base import BaseLayer


class Embedding(BaseLayer):
    """Learnable lookup table for integer token ids."""

    def __init__(self, n_tokens, dim, random_state=None, scale=0.1, padding_idx=None):
        self.n_tokens = int(n_tokens)
        self.dim = int(dim)
        self.random_state = random_state
        self.scale = float(scale)
        self.padding_idx = None if padding_idx is None else int(padding_idx)
        self.W_ = None
        self.dW_ = None
        self.X_ = None
        self._init_params()

    def _init_params(self):
        rng = np.random.default_rng(self.random_state)
        self.W_ = rng.normal(scale=self.scale, size=(self.n_tokens, self.dim))
        if self.padding_idx is not None:
            if not 0 <= self.padding_idx < self.n_tokens:
                raise ValueError("padding_idx must be within [0, n_tokens)")
            self.W_[self.padding_idx] = 0.0

    def forward(self, X):
        X = np.asarray(X)
        if X.size == 0:
            raise ValueError("Expected a non-empty array")
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array of token ids, got shape {X.shape}")
        if not np.issubdtype(X.dtype, np.integer):
            if not np.allclose(X, np.round(X)):
                raise TypeError("Embedding input must contain integer token ids")
            X = np.asarray(np.round(X), dtype=int)
        else:
            X = X.astype(int, copy=False)

        if np.any(X < 0) or np.any(X >= self.n_tokens):
            raise ValueError("token ids must be in [0, n_tokens)")

        self.X_ = X
        return self.W_[X]

    def backward(self, grad):
        grad = np.asarray(grad, dtype=float)
        if self.X_ is None:
            raise RuntimeError("forward must be called before backward")
        if grad.shape != (self.X_.shape[0], self.X_.shape[1], self.dim):
            raise ValueError("Gradient shape does not match embedding output shape")

        self.dW_ = np.zeros_like(self.W_)
        flat_idx = self.X_.ravel()
        flat_grad = grad.reshape(-1, self.dim)
        np.add.at(self.dW_, flat_idx, flat_grad)
        if self.padding_idx is not None:
            self.dW_[self.padding_idx] = 0.0
        return np.zeros_like(self.X_, dtype=float)

    def step(self, lr=None, l2=0.0, optimizer=None, key_prefix="embedding"):
        if self.dW_ is None:
            raise RuntimeError("backward must be called before step")
        if self.padding_idx is not None:
            self.dW_[self.padding_idx] = 0.0
        grad = self.dW_
        if l2:
            grad = grad + l2 * self.W_
        if optimizer is None:
            if lr is None:
                raise ValueError("lr must be provided when optimizer is None")
            self.W_ -= lr * grad
        else:
            self.W_ = optimizer.update(self.W_, grad, f"{key_prefix}.W")
        if self.padding_idx is not None:
            self.W_[self.padding_idx] = 0.0
