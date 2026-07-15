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
        self.W_ = None; self.b_ = None
        self.X_ = None; self.X_shape_ = None; self.cols_ = None; self.dW_ = None; self.db_ = None

    def _init_params(self, in_channels):
        rng = np.random.default_rng(self.random_state)
        s = np.sqrt(2.0 / max(self.k * in_channels, 1))
        self.W_ = rng.normal(scale=s, size=(self.k * in_channels, self.n_filters))
        self.b_ = np.zeros(self.n_filters, dtype=float)

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 3: raise ValueError(f"Expected 3D (N,T,C), got {X.shape}")
        N, T, C = X.shape
        if self.W_ is None: self._init_params(C)
        self.X_ = X; self.X_shape_ = X.shape
        cols, OT = im2col1d(X, self.k, self.stride, self.pad)
        self.cols_ = cols
        out = cols @ self.W_ + self.b_
        return out.reshape(X.shape[0], OT, self.n_filters)

    def backward(self, grad):
        grad = np.asarray(grad, dtype=float)
        N, OT, F = grad.shape
        C = self.X_shape_[2]
        grad_flat = grad.reshape(N * OT, F)
        if self.cols_ is None:
            cols, _ = im2col1d(self.X_, self.k, self.stride, self.pad)
        else:
            cols = self.cols_
        self.dW_ = cols.T @ grad_flat
        self.cols_ = None
        self.db_ = np.sum(grad_flat, axis=0)
        d_cols = grad_flat @ self.W_.T
        return col2im1d(d_cols, self.X_shape_, self.k, self.stride, self.pad)

    def step(self, lr=None, optimizer=None, key_prefix="conv1d"):
        if self.dW_ is None: raise RuntimeError("backward required")
        if optimizer is None: self.W_ -= lr * self.dW_; self.b_ -= lr * self.db_
        else: self.W_ = optimizer.update(self.W_, self.dW_, f"{key_prefix}.W"); self.b_ = optimizer.update(self.b_, self.db_, f"{key_prefix}.b")


class AvgPool1D(BaseLayer):
    def __init__(self, kernel_size=2, stride=None):
        self.k = int(kernel_size); self.s = int(stride) if stride else self.k; self.shape_ = None

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 3: raise ValueError(f"Expected 3D, got {X.shape}")
        N, T, C = X.shape; self.shape_ = X.shape
        OT = (T - self.k) // self.s + 1
        out = np.empty((N, OT, C), dtype=float)
        for i in range(OT):
            out[:, i, :] = np.mean(X[:, i*self.s:i*self.s+self.k, :], axis=1)
        return out

    def backward(self, grad):
        N, T, C = self.shape_; OT = grad.shape[1]; dX = np.zeros(self.shape_, dtype=float)
        for i in range(OT):
            dX[:, i*self.s:i*self.s+self.k, :] += grad[:, i:i+1, :] / self.k
        return dX
