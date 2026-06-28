from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ExponentialFamily(ABC):
    """Base interface for exponential-family distributions."""

    @abstractmethod
    def sufficient_statistics(self, x):
        raise NotImplementedError

    @abstractmethod
    def natural_parameters(self):
        raise NotImplementedError

    @abstractmethod
    def log_base_measure(self, x):
        raise NotImplementedError

    @abstractmethod
    def log_normalizer(self):
        raise NotImplementedError

    def log_prob(self, x):
        x = np.asarray(x, dtype=float)
        stats = self.sufficient_statistics(x)
        eta = self.natural_parameters()
        base = self.log_base_measure(x)
        total = np.zeros_like(base, dtype=float)
        for param, stat in zip(eta, stats):
            contrib = np.asarray(param * stat, dtype=float)
            while contrib.ndim > total.ndim:
                contrib = np.sum(contrib, axis=-1)
            total += contrib
        return total + base - self.log_normalizer()
