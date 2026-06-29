from __future__ import annotations

import numpy as np

from ...base import BaseLayer


class GlobalAveragePooling1D(BaseLayer):
    """Average pool across time axis (N,T,C) -> (N,C)."""
    def __init__(self): self.shape_ = None
    def forward(self, X):
        X = np.asarray(X, dtype=float); self.shape_ = X.shape
        return np.mean(X, axis=1)
    def backward(self, grad):
        grad = np.asarray(grad, dtype=float); N, T, C = self.shape_
        return np.broadcast_to(grad[:, None, :] / T, self.shape_)


class GlobalAveragePooling2D(BaseLayer):
    """Average pool across H,W (N,H,W,C) -> (N,C)."""
    def __init__(self): self.shape_ = None
    def forward(self, X):
        X = np.asarray(X, dtype=float); self.shape_ = X.shape
        return np.mean(X, axis=(1, 2))
    def backward(self, grad):
        grad = np.asarray(grad, dtype=float); N, H, W, C = self.shape_
        return np.broadcast_to(grad[:, None, None, :] / (H*W), self.shape_)


class AvgPool2D(BaseLayer):
    """Average pooling (N,H,W,C) -> (N,OH,OW,C)."""
    def __init__(self, kernel_size=2, stride=None):
        self.k = int(kernel_size); self.s = int(stride) if stride else self.k; self.shape_ = None
    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 4: raise ValueError(f"Expected 4D, got {X.shape}")
        N, H, W, C = X.shape; self.shape_ = X.shape
        OH, OW = (H - self.k)//self.s + 1, (W - self.k)//self.s + 1
        out = np.empty((N, OH, OW, C), dtype=float)
        for i in range(OH):
            for j in range(OW):
                out[:, i, j, :] = np.mean(X[:, i*self.s:i*self.s+self.k, j*self.s:j*self.s+self.k, :], axis=(1,2))
        return out
    def backward(self, grad):
        N, H, W, C = self.shape_; OH, OW = grad.shape[1:3]
        dX = np.zeros(self.shape_, dtype=float); scale = 1.0/(self.k*self.k)
        for i in range(OH):
            for j in range(OW):
                dX[:, i*self.s:i*self.s+self.k, j*self.s:j*self.s+self.k, :] += grad[:, i:i+1, j:j+1, :]*scale
        return dX
