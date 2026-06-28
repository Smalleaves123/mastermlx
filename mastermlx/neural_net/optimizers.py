from __future__ import annotations

import numpy as np


class SGD:
    """Stochastic gradient descent with optional momentum."""

    def __init__(self, lr=0.01, momentum=0.0, nesterov=False):
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self._velocity = {}

    def update(self, param, grad, key):
        if self.momentum == 0.0:
            return param - self.lr * grad
        v = self._velocity.get(key, np.zeros_like(param))
        v = self.momentum * v - self.lr * grad
        self._velocity[key] = v
        if self.nesterov:
            return param + self.momentum * v - self.lr * grad
        return param + v


class RMSProp:
    """RMSProp optimizer."""

    def __init__(self, lr=0.001, rho=0.9, eps=1e-8):
        self.lr = float(lr)
        self.rho = float(rho)
        self.eps = float(eps)
        self._avg_sq = {}

    def update(self, param, grad, key):
        avg_sq = self._avg_sq.get(key, np.zeros_like(param))
        avg_sq = self.rho * avg_sq + (1.0 - self.rho) * (grad ** 2)
        self._avg_sq[key] = avg_sq
        return param - self.lr * grad / (np.sqrt(avg_sq) + self.eps)


class AdaGrad:
    """AdaGrad optimizer with accumulated squared gradients."""

    def __init__(self, lr=0.01, eps=1e-8):
        self.lr = float(lr)
        self.eps = float(eps)
        self._G = {}

    def update(self, param, grad, key):
        G = self._G.get(key, np.zeros_like(param))
        G += grad ** 2
        self._G[key] = G
        return param - self.lr * grad / (np.sqrt(G) + self.eps)


class Adam:
    """Adam optimizer."""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self._m = {}
        self._v = {}
        self._t = 0

    def update(self, param, grad, key):
        self._t += 1
        m = self._m.get(key, np.zeros_like(param))
        v = self._v.get(key, np.zeros_like(param))
        m = self.beta1 * m + (1.0 - self.beta1) * grad
        v = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)
        self._m[key] = m
        self._v[key] = v
        m_hat = m / (1.0 - self.beta1 ** self._t)
        v_hat = v / (1.0 - self.beta2 ** self._t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
