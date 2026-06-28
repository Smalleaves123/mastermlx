from __future__ import annotations

import numpy as np


class BernoulliThompsonSampling:
    """Thompson sampling for Bernoulli rewards."""

    def __init__(self, n_arms, alpha=1.0, beta=1.0, random_state=None):
        self.n_arms = int(n_arms)
        self.alpha_prior = float(alpha)
        self.beta_prior = float(beta)
        self.random_state = random_state
        self.alpha_ = np.full(self.n_arms, self.alpha_prior, dtype=float)
        self.beta_ = np.full(self.n_arms, self.beta_prior, dtype=float)
        self.total_reward_ = 0.0
        self._rng = np.random.default_rng(random_state)

    def select_arm(self):
        samples = self._rng.beta(self.alpha_, self.beta_)
        return int(np.argmax(samples))

    def update(self, arm, reward):
        arm = int(arm)
        reward = float(reward)
        clipped = 1.0 if reward > 0 else 0.0
        self.alpha_[arm] += clipped
        self.beta_[arm] += 1.0 - clipped
        self.total_reward_ += reward


BernThompson = BernoulliThompsonSampling
