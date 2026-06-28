from __future__ import annotations

import numpy as np

from ...base import BaseLayer
from ...utils.validation import check_2d_array


class ReLU(BaseLayer):
    def __init__(self):
        self.X_ = None

    def forward(self, X):
        X = check_2d_array(X)
        self.X_ = X
        return np.maximum(0.0, X)

    def backward(self, grad):
        grad = check_2d_array(grad)
        return grad * (self.X_ > 0.0)


class Tanh(BaseLayer):
    def __init__(self):
        self.out_ = None

    def forward(self, X):
        X = check_2d_array(X)
        self.out_ = np.tanh(X)
        return self.out_

    def backward(self, grad):
        grad = check_2d_array(grad)
        return grad * (1.0 - self.out_ ** 2)


class LeakyReLU(BaseLayer):
    def __init__(self, alpha=0.01):
        self.alpha = float(alpha)
        self.X_ = None

    def forward(self, X):
        X = check_2d_array(X)
        self.X_ = X
        return np.where(X > 0, X, self.alpha * X)

    def backward(self, grad):
        grad = check_2d_array(grad)
        return grad * np.where(self.X_ > 0, 1.0, self.alpha)


class GELU(BaseLayer):
    def __init__(self):
        self.X_ = None

    def forward(self, X):
        X = check_2d_array(X)
        self.X_ = X
        cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (X + 0.044715 * X ** 3)))
        return X * cdf

    def backward(self, grad):
        grad = check_2d_array(grad)
        x = self.X_
        sqrt_2_pi = np.sqrt(2.0 / np.pi)
        inner = sqrt_2_pi * (x + 0.044715 * x ** 3)
        cdf = 0.5 * (1.0 + np.tanh(inner))
        pdf = 0.5 * sqrt_2_pi * (1.0 + 3.0 * 0.044715 * x ** 2) * (1.0 - np.tanh(inner) ** 2)
        return grad * (cdf + x * pdf)


class Sigmoid(BaseLayer):
    def __init__(self):
        self.out_ = None

    def forward(self, X):
        X = check_2d_array(X)
        pos = X >= 0
        neg = ~pos
        out = np.empty_like(X, dtype=float)
        out[pos] = 1.0 / (1.0 + np.exp(-X[pos]))
        exp_x = np.exp(X[neg])
        out[neg] = exp_x / (1.0 + exp_x)
        self.out_ = out
        return out

    def backward(self, grad):
        grad = check_2d_array(grad)
        return grad * self.out_ * (1.0 - self.out_)
