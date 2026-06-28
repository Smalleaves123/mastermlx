from __future__ import annotations

import numpy as np

from ...base import BaseLayer


def _im2col(X, kh, kw, stride=1, pad=0):
    """Convert image patches to columns for fast convolution via matmul.
    Input:  (N, H, W, C)  →  Output: (N*OH*OW, KH*KW*C)
    """
    N, H, W, C = X.shape
    if pad > 0:
        X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    else:
        X_pad = X
    OH = (H + 2 * pad - kh) // stride + 1
    OW = (W + 2 * pad - kw) // stride + 1
    cols = np.empty((N, OH, OW, kh, kw, C), dtype=X.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, :, i, j, :] = X_pad[:, i:i + OH * stride:stride, j:j + OW * stride:stride, :]
    return cols.reshape(N * OH * OW, kh * kw * C), OH, OW


def _col2im(cols, shape, kh, kw, stride=1, pad=0):
    """Reverse of im2col: accumulate column gradients back to image shape."""
    N, H, W, C = shape
    OH = (H + 2 * pad - kh) // stride + 1
    OW = (W + 2 * pad - kw) // stride + 1
    cols_6d = cols.reshape(N, OH, OW, kh, kw, C)
    dX = np.zeros((N, H + 2 * pad, W + 2 * pad, C), dtype=cols.dtype)
    for i in range(kh):
        for j in range(kw):
            dX[:, i:i + OH * stride:stride, j:j + OW * stride:stride, :] += cols_6d[:, :, :, i, j, :]
    if pad > 0:
        return dX[:, pad:-pad, pad:-pad, :]
    return dX


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
        cols, oh, ow = _im2col(X, kh, kw, self.stride, self.pad)
        out = cols @ self.W_ + self.b_
        return out.reshape(N, oh, ow, self.n_filters)

    def backward(self, grad):
        grad = np.asarray(grad, dtype=float)
        N, OH, OW, F = grad.shape
        kh, kw = self.kernel_size
        cols, _, _ = _im2col(self.X_, kh, kw, self.stride, self.pad)
        grad_flat = grad.reshape(N * OH * OW, F)
        self.dW_ = cols.T @ grad_flat
        self.db_ = np.sum(grad_flat, axis=0)
        d_cols = grad_flat @ self.W_.T
        return _col2im(d_cols, self.X_shape_, kh, kw, self.stride, self.pad)

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
        self.mask_ = None

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 4:
            raise ValueError(f"Expected 4D input (N,H,W,C), got {X.shape}")
        N, H, W, C = X.shape
        self.X_shape_ = X.shape
        k, s = self.k, self.stride
        OH = (H - k) // s + 1
        OW = (W - k) // s + 1
        out = np.empty((N, OH, OW, C), dtype=float)
        self.mask_ = np.zeros(X.shape, dtype=bool)
        for i in range(OH):
            for j in range(OW):
                patch = X[:, i*s:i*s+k, j*s:j*s+k, :]
                flat = patch.reshape(N, k * k, C)
                idx = np.argmax(flat, axis=1)
                out[:, i, j, :] = np.take_along_axis(flat, idx[:, None, :], axis=1).squeeze(1)
                # Build mask for backward
                bi, bj = np.unravel_index(idx.ravel(), (k, k))
                for b in range(N):
                    for c in range(C):
                        self.mask_[b, i*s+bi[b*C+c], j*s+bj[b*C+c], c] = True
        return out

    def backward(self, grad):
        grad = np.asarray(grad, dtype=float)
        N, OH, OW, C = grad.shape
        k, s = self.k, self.stride
        dX = np.zeros(self.X_shape_, dtype=float)
        for i in range(OH):
            for j in range(OW):
                dX[:, i*s:i*s+k, j*s:j*s+k, :] += grad[:, i:i+1, j:j+1, :] * \
                    self.mask_[:, i*s:i*s+k, j*s:j*s+k, :]
        return dX


class Flatten(BaseLayer):
    """Flatten spatial dims to a 1D feature vector."""

    def __init__(self):
        self.shape_ = None

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        self.shape_ = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grad):
        return np.asarray(grad, dtype=float).reshape(self.shape_)
