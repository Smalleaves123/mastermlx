from __future__ import annotations

import numpy as np


class UCBBandit:
    """Upper confidence bound bandit."""

    def __init__(self, n_arms, c=2.0):
        self.n_arms = int(n_arms)
        self.c = float(c)
        self.q_values_ = np.zeros(self.n_arms, dtype=float)
        self.counts_ = np.zeros(self.n_arms, dtype=int)
        self.t_ = 0
        self.total_reward_ = 0.0

    def select_arm(self):
        for arm in range(self.n_arms):
            if self.counts_[arm] == 0:
                return arm
        bonus = self.c * np.sqrt(np.log(max(self.t_, 1)) / self.counts_)
        return int(np.argmax(self.q_values_ + bonus))

    def update(self, arm, reward):
        arm = int(arm)
        reward = float(reward)
        self.t_ += 1
        self.counts_[arm] += 1
        step = 1.0 / self.counts_[arm]
        self.q_values_[arm] += step * (reward - self.q_values_[arm])
        self.total_reward_ += reward
