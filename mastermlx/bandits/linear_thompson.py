from __future__ import annotations

import numpy as np

from ..utils import check_1d_array


class LinearThompsonBandit:
    """Linear Thompson sampling contextual bandit."""

    def __init__(self, n_arms, n_features, v=1.0, random_state=None):
        self.n_arms = int(n_arms)
        self.n_features = int(n_features)
        self.v = float(v)
        self.random_state = random_state
        if self.n_arms < 1:
            raise ValueError("n_arms must be at least 1")
        if self.n_features < 1:
            raise ValueError("n_features must be at least 1")
        if self.v <= 0.0:
            raise ValueError("v must be positive")

        self.A_ = np.repeat(np.eye(self.n_features)[None, :, :], self.n_arms, axis=0)
        self.b_ = np.zeros((self.n_arms, self.n_features), dtype=float)
        self.counts_ = np.zeros(self.n_arms, dtype=int)
        self.total_reward_ = 0.0
        self._rng = np.random.default_rng(random_state)

    def _validate_context(self, context):
        context = check_1d_array(context, name="context").astype(float)
        if context.shape[0] != self.n_features:
            raise ValueError("context has a different number of features than expected")
        return context

    def _posterior_mean_cov(self, arm):
        mean = np.linalg.solve(self.A_[arm], self.b_[arm])
        cov = (self.v**2) * np.linalg.inv(self.A_[arm])
        return mean, cov

    def predict_reward(self, context):
        context = self._validate_context(context)
        rewards = np.empty(self.n_arms, dtype=float)
        for arm in range(self.n_arms):
            mean, _ = self._posterior_mean_cov(arm)
            rewards[arm] = context @ mean
        return rewards

    def sample_reward(self, context):
        context = self._validate_context(context)
        samples = np.empty(self.n_arms, dtype=float)
        for arm in range(self.n_arms):
            mean, cov = self._posterior_mean_cov(arm)
            theta = self._rng.multivariate_normal(mean, cov)
            samples[arm] = context @ theta
        return samples

    def select_arm(self, context, return_samples=False):
        samples = self.sample_reward(context)
        arm = int(np.argmax(samples))
        if return_samples:
            return arm, samples
        return arm

    def update(self, arm, context, reward):
        arm = int(arm)
        context = self._validate_context(context)
        reward = float(reward)
        self.A_[arm] += np.outer(context, context)
        self.b_[arm] += reward * context
        self.counts_[arm] += 1
        self.total_reward_ += reward


LinThompson = LinearThompsonBandit
