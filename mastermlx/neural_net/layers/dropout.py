from __future__ import annotations

import numpy as np

from ...base import BaseLayer
from ...utils.validation import check_2d_array


class Dropout(BaseLayer):
    """Inverted dropout layer."""

    def __init__(self, rate=0.5, random_state=None):
        self.rate = float(rate)
        self.random_state = random_state
        self.training = True
        self.mask_ = None
        self._rng = np.random.default_rng(random_state)
        if not 0.0 <= self.rate < 1.0:
            raise ValueError("rate must be in [0, 1)")

    def train(self, mode=True):
        self.training = bool(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, X):
        X = check_2d_array(X)
        if not self.training or self.rate == 0.0:
            self.mask_ = None
            return X
        keep_prob = 1.0 - self.rate
        self.mask_ = (self._rng.random(X.shape) < keep_prob).astype(float) / keep_prob
        return X * self.mask_

    def backward(self, grad):
        grad = check_2d_array(grad)
        if not self.training or self.mask_ is None:
            return grad
        return grad * self.mask_
