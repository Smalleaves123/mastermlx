from __future__ import annotations

import numpy as np

from ...base import BaseLayer
from ...utils.validation import check_2d_array


class BatchNorm(BaseLayer):
    """Batch normalization for 2D inputs."""

    def __init__(self, n_features, momentum=0.9, eps=1e-5, random_state=None):
        self.n_features = int(n_features)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.random_state = random_state
        self.training = True
        self.gamma_ = np.ones(self.n_features)
        self.beta_ = np.zeros(self.n_features)
        self.running_mean_ = np.zeros(self.n_features)
        self.running_var_ = np.ones(self.n_features)
        self.X_ = None
        self.X_centered_ = None
        self.X_norm_ = None
        self.std_inv_ = None
        self.dgamma_ = None
        self.dbeta_ = None

    def train(self, mode=True):
        self.training = bool(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, X):
        X = check_2d_array(X)
        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")

        if self.training:
            mean = np.mean(X, axis=0)
            var = np.var(X, axis=0)
            self.X_ = X
            self.X_centered_ = X - mean
            self.std_inv_ = 1.0 / np.sqrt(var + self.eps)
            self.X_norm_ = self.X_centered_ * self.std_inv_
            self.running_mean_ = self.momentum * self.running_mean_ + (1.0 - self.momentum) * mean
            self.running_var_ = self.momentum * self.running_var_ + (1.0 - self.momentum) * var
            return self.gamma_ * self.X_norm_ + self.beta_

        X_norm = (X - self.running_mean_) / np.sqrt(self.running_var_ + self.eps)
        return self.gamma_ * X_norm + self.beta_

    def backward(self, grad):
        grad = check_2d_array(grad)
        if not self.training or self.X_ is None:
            return grad

        X = self.X_
        centered = self.X_centered_
        norm = self.X_norm_
        std_inv = self.std_inv_
        if X is None or centered is None or norm is None or std_inv is None:
            raise RuntimeError("forward must be called before backward")
        n = X.shape[0]
        self.dbeta_ = np.sum(grad, axis=0)
        self.dgamma_ = np.sum(grad * norm, axis=0)

        dx_norm = grad * self.gamma_
        dvar = np.sum(dx_norm * centered * -0.5 * std_inv**3, axis=0)
        dmean = np.sum(dx_norm * -std_inv, axis=0) + dvar * np.mean(-2.0 * centered, axis=0)
        dx = dx_norm * std_inv + dvar * 2.0 * centered / n + dmean / n
        return dx

    def step(self, lr=None, optimizer=None, key_prefix="batchnorm"):
        if self.dgamma_ is None or self.dbeta_ is None:
            raise RuntimeError("backward must be called before step")
        if optimizer is None:
            if lr is None:
                raise ValueError("lr must be provided when optimizer is None")
            self.gamma_ -= lr * self.dgamma_
            self.beta_ -= lr * self.dbeta_
            return
        self.gamma_ = optimizer.update(self.gamma_, self.dgamma_, f"{key_prefix}.gamma")
        self.beta_ = optimizer.update(self.beta_, self.dbeta_, f"{key_prefix}.beta")
