from __future__ import annotations

import numpy as np

from ...accel.conv1d_ops import col2im1d, im2col1d
from ...base import BaseLayer


class Conv1D(BaseLayer):
    """1D convolution for sequences (N, T, C). Uses im2col + matmul."""

    def __init__(self, n_filters, kernel_size, stride=1, pad=0, random_state=None):
        self.n_filters = int(n_filters)
        self.k = int(kernel_size)
        self.stride = int(stride)
        self.pad = int(pad)
        self.random_state = random_state
        self.W_ = None
        self.b_ = None
        self.X_ = None
        self.X_shape_ = None
        self.cols_ = None
        self.dW_ = None
        self.db_ = None

    def _init_params(self, in_channels):
        rng = np.random.default_rng(self.random_state)
        s = np.sqrt(2.0 / max(self.k * in_channels, 1))
        self.W_ = rng.normal(scale=s, size=(self.k * in_channels, self.n_filters))
        self.b_ = np.zeros(self.n_filters, dtype=float)

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError(f"Expected 3D (N,T,C), got {X.shape}")
        N, T, C = X.shape
        if self.W_ is None:
            self._init_params(C)
        self.X_ = X
        self.X_shape_ = X.shape
        cols, OT = im2col1d(X, self.k, self.stride, self.pad)
        self.cols_ = cols
        out = cols @ self.W_ + self.b_
        return out.reshape(X.shape[0], OT, self.n_filters)

    def backward(self, grad):
        grad = np.asarray(grad, dtype=float)
        N, OT, F = grad.shape
        X = self.X_
        X_shape = self.X_shape_
        W = self.W_
        if X is None or X_shape is None or W is None:
            raise RuntimeError("forward must be called before backward")
        grad_flat = grad.reshape(N * OT, F)
        if self.cols_ is None:
            cols, _ = im2col1d(X, self.k, self.stride, self.pad)
        else:
            cols = self.cols_
        self.dW_ = cols.T @ grad_flat
        self.cols_ = None
        self.db_ = np.sum(grad_flat, axis=0)
        d_cols = grad_flat @ W.T
        return col2im1d(d_cols, X_shape, self.k, self.stride, self.pad)

    def step(self, lr=None, optimizer=None, key_prefix="conv1d"):
        W = self.W_
        b = self.b_
        dW = self.dW_
        db = self.db_
        if W is None or b is None or dW is None or db is None:
            raise RuntimeError("backward required")
        if optimizer is None:
            self.W_ = W - lr * dW
            self.b_ = b - lr * db
        else:
            self.W_ = optimizer.update(W, dW, f"{key_prefix}.W")
            self.b_ = optimizer.update(b, db, f"{key_prefix}.b")


class AvgPool1D(BaseLayer):
    def __init__(self, kernel_size=2, stride=None):
        self.k = int(kernel_size)
        self.s = int(stride) if stride else self.k
        self.shape_ = None

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError(f"Expected 3D, got {X.shape}")
        N, T, C = X.shape
        self.shape_ = X.shape
        OT = (T - self.k) // self.s + 1
        out = np.empty((N, OT, C), dtype=float)
        for i in range(OT):
            out[:, i, :] = np.mean(X[:, i * self.s : i * self.s + self.k, :], axis=1)
        return out

    def backward(self, grad):
        shape = self.shape_
        if shape is None:
            raise RuntimeError("forward must be called before backward")
        N, T, C = shape
        OT = grad.shape[1]
        dX = np.zeros(shape, dtype=float)
        for i in range(OT):
            dX[:, i * self.s : i * self.s + self.k, :] += grad[:, i : i + 1, :] / self.k
        return dX
