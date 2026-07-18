from __future__ import annotations

import numpy as np

from ...base import BaseLayer
from ...utils.validation import check_2d_array


class Dense(BaseLayer):
    """Fully connected layer with gradient storage."""

    def __init__(self, n_inputs, n_outputs, random_state=None, init="xavier"):
        self.n_inputs = int(n_inputs)
        self.n_outputs = int(n_outputs)
        self.random_state = random_state
        self.init = init
        self.W_ = None
        self.b_ = None
        self.X_ = None
        self.dW_ = None
        self.db_ = None
        self._init_params()

    def _init_params(self):
        rng = np.random.default_rng(self.random_state)
        if self.init == "he":
            scale = np.sqrt(2.0 / max(self.n_inputs, 1))
        else:
            scale = np.sqrt(1.0 / max(self.n_inputs, 1))
        self.W_ = rng.normal(scale=scale, size=(self.n_inputs, self.n_outputs))
        self.b_ = np.zeros(self.n_outputs)

    def forward(self, X):
        X = check_2d_array(X)
        if X.shape[1] != self.n_inputs:
            raise ValueError(f"Expected {self.n_inputs} features, got {X.shape[1]}")
        self.X_ = X
        return X @ self.W_ + self.b_

    def backward(self, grad):
        grad = check_2d_array(grad)
        X = self.X_
        W = self.W_
        if X is None or W is None:
            raise RuntimeError("forward must be called before backward")
        self.dW_ = X.T @ grad
        self.db_ = np.sum(grad, axis=0)
        return grad @ W.T

    def step(self, lr=None, l2=0.0, optimizer=None, key_prefix="dense"):
        if self.dW_ is None or self.db_ is None:
            raise RuntimeError("backward must be called before step")
        grad_w = self.dW_
        if l2:
            grad_w = grad_w + l2 * self.W_
        if optimizer is None:
            if lr is None:
                raise ValueError("lr must be provided when optimizer is None")
            self.W_ -= lr * grad_w
            self.b_ -= lr * self.db_
            return
        self.W_ = optimizer.update(self.W_, grad_w, f"{key_prefix}.W")
        self.b_ = optimizer.update(self.b_, self.db_, f"{key_prefix}.b")
