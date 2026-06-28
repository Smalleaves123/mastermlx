from __future__ import annotations

import numpy as np

from ..utils import check_1d_array


class LinUCBBandit:
    """Linear UCB contextual bandit."""

    def __init__(self, n_arms, n_features, alpha=1.0):
        self.n_arms = int(n_arms)
        self.n_features = int(n_features)
        self.alpha = float(alpha)
        if self.n_arms < 1:
            raise ValueError("n_arms must be at least 1")
        if self.n_features < 1:
            raise ValueError("n_features must be at least 1")
        if self.alpha < 0.0:
            raise ValueError("alpha must be non-negative")

        self.A_ = np.repeat(np.eye(self.n_features)[None, :, :], self.n_arms, axis=0)
        self.b_ = np.zeros((self.n_arms, self.n_features), dtype=float)
        self.counts_ = np.zeros(self.n_arms, dtype=int)
        self.total_reward_ = 0.0

    def _validate_context(self, context):
        context = check_1d_array(context, name="context").astype(float)
        if context.shape[0] != self.n_features:
            raise ValueError("context has a different number of features than expected")
        return context

    def _arm_theta(self, arm):
        return np.linalg.solve(self.A_[arm], self.b_[arm])

    def predict_reward(self, context):
        context = self._validate_context(context)
        rewards = np.empty(self.n_arms, dtype=float)
        for arm in range(self.n_arms):
            rewards[arm] = context @ self._arm_theta(arm)
        return rewards

    def select_arm(self, context, return_scores=False):
        context = self._validate_context(context)
        scores = np.empty(self.n_arms, dtype=float)
        for arm in range(self.n_arms):
            a_inv_context = np.linalg.solve(self.A_[arm], context)
            mean = context @ self._arm_theta(arm)
            bonus = self.alpha * np.sqrt(context @ a_inv_context)
            scores[arm] = mean + bonus
        best_arm = int(np.argmax(scores))
        if return_scores:
            return best_arm, scores
        return best_arm

    def update(self, arm, context, reward):
        arm = int(arm)
        context = self._validate_context(context)
        reward = float(reward)
        self.A_[arm] += np.outer(context, context)
        self.b_[arm] += reward * context
        self.counts_[arm] += 1
        self.total_reward_ += reward
