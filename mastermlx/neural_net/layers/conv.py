from __future__ import annotations

import numpy as np

from ...accel.cnn_ops import col2im, im2col, maxpool_backward, maxpool_forward
from ...base import BaseLayer


class Conv2D(BaseLayer):
    """2D convolution layer with im2col + matmul."""

    def __init__(self, n_filters, kernel_size, stride=1, pad=0, random_state=None):
        self.n_filters = int(n_filters)
        self.kernel_size = (int(kernel_size), int(kernel_size)) if isinstance(kernel_size, int) else tuple(kernel_size)
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
        kh, kw = self.kernel_size
        rng = np.random.default_rng(self.random_state)
        scale = np.sqrt(2.0 / max(kh * kw * in_channels, 1))
        self.W_ = rng.normal(scale=scale, size=(kh * kw * in_channels, self.n_filters))
        self.b_ = np.zeros(self.n_filters, dtype=float)

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 4:
            raise ValueError(f"Expected 4D input (N,H,W,C), got {X.shape}")
        N, H, W, C = X.shape
        kh, kw = self.kernel_size
        if self.W_ is None:
            self._init_params(C)
        self.X_shape_ = X.shape
        self.X_ = X
        cols, oh, ow = im2col(X, kh, kw, self.stride, self.pad)
        self.cols_ = cols
        out = cols @ self.W_ + self.b_
        return out.reshape(N, oh, ow, self.n_filters)

    def backward(self, grad):
        grad = np.asarray(grad, dtype=float)
        if self.X_shape_ is None or self.W_ is None:
            raise RuntimeError("forward must be called before backward")
        if grad.ndim != 4:
            raise ValueError(f"Expected 4D gradient (N,OH,OW,F), got {grad.shape}")
        N, H, W, C = self.X_shape_
        kh, kw = self.kernel_size
        oh = (H + 2 * self.pad - kh) // self.stride + 1
        ow = (W + 2 * self.pad - kw) // self.stride + 1
        expected = (N, oh, ow, self.n_filters)
        if grad.shape != expected:
            raise ValueError(f"Expected gradient shape {expected}, got {grad.shape}")
        N, OH, OW, F = grad.shape
        grad_flat = grad.reshape(N * OH * OW, F)
        if self.cols_ is None:
            cols, _, _ = im2col(self.X_, kh, kw, self.stride, self.pad)
        else:
            cols = self.cols_
        self.dW_ = cols.T @ grad_flat
        self.cols_ = None
        self.db_ = np.sum(grad_flat, axis=0)
        d_cols = grad_flat @ self.W_.T
        return col2im(d_cols, self.X_shape_, kh, kw, self.stride, self.pad)

    def step(self, lr=None, optimizer=None, key_prefix="conv"):
        if self.dW_ is None or self.db_ is None:
            raise RuntimeError("backward must be called before step")
        if optimizer is None:
            if lr is None:
                raise ValueError("lr required when optimizer is None")
            self.W_ -= lr * self.dW_
            self.b_ -= lr * self.db_
            return
        self.W_ = optimizer.update(self.W_, self.dW_, f"{key_prefix}.W")
        self.b_ = optimizer.update(self.b_, self.db_, f"{key_prefix}.b")


class MaxPool2D(BaseLayer):
    """2D max pooling layer."""

    def __init__(self, kernel_size=2, stride=None):
        self.k = int(kernel_size)
        self.stride = int(stride) if stride else self.k
        self.X_shape_ = None
        self.argmax_ = None

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 4:
            raise ValueError(f"Expected 4D input (N,H,W,C), got {X.shape}")
        N, H, W, C = X.shape
        self.X_shape_ = X.shape
        k, s = self.k, self.stride
        out, self.argmax_ = maxpool_forward(X, k, s)
        return out

    def backward(self, grad):
        grad = np.asarray(grad, dtype=float)
        if self.argmax_ is None or self.X_shape_ is None:
            raise RuntimeError("forward must be called before backward")
        return maxpool_backward(grad, self.argmax_, self.X_shape_, self.k, self.stride)


class Flatten(BaseLayer):
    """Flatten spatial dims to a 1D feature vector."""

    def __init__(self):
        self.shape_ = None

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        self.shape_ = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grad):
        if self.shape_ is None:
            raise RuntimeError("forward must be called before backward")
        return np.asarray(grad, dtype=float).reshape(self.shape_)
