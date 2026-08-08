from __future__ import annotations

import numpy as np


class Exp3Bandit:
    """EXP3 bandit for adversarial reward settings."""

    def __init__(self, n_arms, gamma=0.1, random_state=None):
        self.n_arms = int(n_arms)
        self.gamma = float(gamma)
        self.random_state = random_state
        if self.n_arms < 1:
            raise ValueError("n_arms must be at least 1")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")

        self.weights_ = np.ones(self.n_arms, dtype=float)
        self.counts_ = np.zeros(self.n_arms, dtype=int)
        self.total_reward_ = 0.0
        self._rng = np.random.default_rng(random_state)

    def arm_probabilities(self):
        weight_sum = np.sum(self.weights_)
        exploit = (1.0 - self.gamma) * (self.weights_ / weight_sum)
        explore = self.gamma / self.n_arms
        return exploit + explore

    def select_arm(self, return_probs=False):
        probs = self.arm_probabilities()
        arm = int(self._rng.choice(self.n_arms, p=probs))
        if return_probs:
            return arm, probs
        return arm

    def update(self, arm, reward, probs=None):
        arm = int(arm)
        reward = float(reward)
        if arm < 0 or arm >= self.n_arms:
            raise ValueError("arm must be in [0, n_arms)")
        if not np.isfinite(reward) or not 0.0 <= reward <= 1.0:
            raise ValueError("EXP3 reward must be finite and in [0, 1]")
        if probs is None:
            probs = self.arm_probabilities()
        probs = np.asarray(probs, dtype=float)
        if probs.shape != (self.n_arms,):
            raise ValueError("probs must have shape (n_arms,)")
        if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
            raise ValueError("probs must contain finite non-negative values")
        if not np.isclose(np.sum(probs), 1.0):
            raise ValueError("probs must sum to 1")
        if probs[arm] <= 0.0:
            raise ValueError("selected arm probability must be positive")

        reward_hat = reward / probs[arm]
        self.weights_[arm] *= np.exp((self.gamma * reward_hat) / self.n_arms)
        self.counts_[arm] += 1
        self.total_reward_ += reward
