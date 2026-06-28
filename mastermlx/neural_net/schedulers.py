"""Learning rate schedulers."""

from __future__ import annotations

import numpy as np


class StepLR:
    """Multiply lr by gamma every step_size epochs."""

    def __init__(self, optimizer, step_size=30, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = int(step_size)
        self.gamma = float(gamma)
        self._epoch = 0
        self._base_lr = getattr(optimizer, 'lr', 0.01)

    def step(self):
        self._epoch += 1
        if self._epoch % self.step_size == 0:
            self.optimizer.lr = self._base_lr * (self.gamma ** (self._epoch // self.step_size))


class CosineLR:
    """Cosine annealing: lr = lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi*t/T))."""

    def __init__(self, optimizer, T_max=100, eta_min=0.0):
        self.optimizer = optimizer
        self.T_max = int(T_max)
        self.eta_min = float(eta_min)
        self._base_lr = getattr(optimizer, 'lr', 0.01)
        self._epoch = 0

    def step(self):
        self._epoch += 1
        t = min(self._epoch, self.T_max)
        cos = np.cos(np.pi * t / self.T_max)
        self.optimizer.lr = self.eta_min + 0.5 * (self._base_lr - self.eta_min) * (1.0 + cos)


class ReduceLROnPlateau:
    """Reduce lr when a metric has stopped improving."""

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = float(factor)
        self.patience = int(patience)
        self.threshold = float(threshold)
        self.best = np.inf if mode == 'min' else -np.inf
        self._wait = 0

    def step(self, metric):
        improved = (self.mode == 'min' and metric < self.best - self.threshold) or \
                   (self.mode == 'max' and metric > self.best + self.threshold)
        if improved:
            self.best = metric
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                self.optimizer.lr *= self.factor
                self._wait = 0
