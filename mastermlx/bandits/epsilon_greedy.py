from __future__ import annotations

import numpy as np


class EpsilonGreedyBandit:
    """Incremental epsilon-greedy multi-armed bandit."""

    def __init__(self, n_arms, epsilon=0.1, initial_value=0.0, random_state=None):
        self.n_arms = int(n_arms)
        self.epsilon = float(epsilon)
        self.initial_value = float(initial_value)
        self.random_state = random_state
        self.q_values_ = np.full(self.n_arms, self.initial_value, dtype=float)
        self.counts_ = np.zeros(self.n_arms, dtype=int)
        self.total_reward_ = 0.0
        self._rng = np.random.default_rng(random_state)

    def select_arm(self):
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_arms))
        return int(np.argmax(self.q_values_))

    def update(self, arm, reward):
        arm = int(arm)
        reward = float(reward)
        self.counts_[arm] += 1
        step = 1.0 / self.counts_[arm]
        self.q_values_[arm] += step * (reward - self.q_values_[arm])
        self.total_reward_ += reward


EpsGreedyBandit = EpsilonGreedyBandit
