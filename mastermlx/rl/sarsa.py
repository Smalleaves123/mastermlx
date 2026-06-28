from __future__ import annotations

import numpy as np


class SARSAAgent:
    """Tabular SARSA (State-Action-Reward-State-Action) agent."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1, random_state=None):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.random_state = random_state
        self.q_table_ = np.zeros((self.n_states, self.n_actions), dtype=float)
        self._rng = np.random.default_rng(random_state)

    def select_action(self, state):
        state = int(state)
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        return int(np.argmax(self.q_table_[state]))

    def update(self, state, action, reward, next_state, next_action, done=False):
        state = int(state)
        action = int(action)
        next_state = int(next_state)
        next_action = int(next_action)
        reward = float(reward)
        target = reward
        if not done:
            target += self.gamma * self.q_table_[next_state, next_action]
        td = target - self.q_table_[state, action]
        self.q_table_[state, action] += self.alpha * td

    def value_function(self):
        return np.max(self.q_table_, axis=1)
