from __future__ import annotations

import numpy as np


class SoftmaxBandit:
    """Boltzmann exploration bandit with incremental value updates."""

    def __init__(self, n_arms, temperature=0.1, initial_value=0.0, random_state=None):
        self.n_arms = int(n_arms)
        self.temperature = float(temperature)
        self.initial_value = float(initial_value)
        self.random_state = random_state
        self.q_values_ = np.full(self.n_arms, self.initial_value, dtype=float)
        self.counts_ = np.zeros(self.n_arms, dtype=int)
        self.total_reward_ = 0.0
        self._rng = np.random.default_rng(random_state)

    def arm_probabilities(self):
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        scaled = self.q_values_ / self.temperature
        scaled = scaled - np.max(scaled)
        probs = np.exp(scaled)
        return probs / np.sum(probs)

    def select_arm(self):
        probs = self.arm_probabilities()
        return int(self._rng.choice(self.n_arms, p=probs))

    def update(self, arm, reward):
        arm = int(arm)
        reward = float(reward)
        self.counts_[arm] += 1
        step = 1.0 / self.counts_[arm]
        self.q_values_[arm] += step * (reward - self.q_values_[arm])
        self.total_reward_ += reward
