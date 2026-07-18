from __future__ import annotations

import numpy as np

from ...base import BaseLayer


class LayerNorm(BaseLayer):
    """Layer normalization over the last axis."""

    def __init__(self, n_features, eps=1e-5):
        self.n_features = int(n_features)
        self.eps = float(eps)
        self.gamma_ = np.ones(n_features, dtype=float)
        self.beta_ = np.zeros(n_features, dtype=float)
        self.X_ = None
        self.norm_ = None
        self.std_ = None
        self.dgamma_ = None
        self.dbeta_ = None

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        self.X_ = X
        mu = np.mean(X, axis=-1, keepdims=True)
        var = np.var(X, axis=-1, keepdims=True)
        self.std_ = np.sqrt(var + self.eps)
        self.norm_ = (X - mu) / self.std_
        return self.norm_ * self.gamma_ + self.beta_

    def backward(self, grad):
        grad = np.asarray(grad, dtype=float)
        X = self.X_
        norm = self.norm_
        std = self.std_
        if X is None or norm is None or std is None:
            raise RuntimeError("forward must be called before backward")
        D = X.shape[-1]
        dx_norm = grad * self.gamma_
        self.dgamma_ = np.sum(grad * norm, axis=tuple(range(grad.ndim - 1)))
        self.dbeta_ = np.sum(grad, axis=tuple(range(grad.ndim - 1)))

        centered = X - np.mean(X, axis=-1, keepdims=True)
        dvar = np.sum(dx_norm * centered, axis=-1, keepdims=True) * (-0.5) * std ** (-3)
        dmu = np.sum(dx_norm * (-1.0 / std), axis=-1, keepdims=True) + dvar * np.mean(
            -2.0 * centered, axis=-1, keepdims=True
        )
        return dx_norm / std + dvar * 2.0 * centered / D + dmu / D

    def step(self, lr=None, optimizer=None, key_prefix="ln"):
        if self.dgamma_ is None:
            raise RuntimeError("backward must be called before step")
        if optimizer is None:
            self.gamma_ -= lr * self.dgamma_
            self.beta_ -= lr * self.dbeta_
            return
        self.gamma_ = optimizer.update(self.gamma_, self.dgamma_, f"{key_prefix}.gamma")
        self.beta_ = optimizer.update(self.beta_, self.dbeta_, f"{key_prefix}.beta")
