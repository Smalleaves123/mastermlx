from __future__ import annotations

import numpy as np


class DoubleQLearningAgent:
    """Double Q-learning — uses two Q-tables to reduce overestimation bias."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1, random_state=None):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.random_state = random_state
        self.q_a_ = np.zeros((self.n_states, self.n_actions), dtype=float)
        self.q_b_ = np.zeros((self.n_states, self.n_actions), dtype=float)
        self._rng = np.random.default_rng(random_state)

    def select_action(self, state):
        state = int(state)
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        return int(np.argmax(self.q_a_[state] + self.q_b_[state]))

    def update(self, state, action, reward, next_state, done=False):
        state = int(state)
        action = int(action)
        next_state = int(next_state)
        reward = float(reward)
        if self._rng.random() < 0.5:
            best_a = int(np.argmax(self.q_a_[next_state]))
            target = reward
            if not done:
                target += self.gamma * self.q_b_[next_state, best_a]
            self.q_a_[state, action] += self.alpha * (target - self.q_a_[state, action])
        else:
            best_a = int(np.argmax(self.q_b_[next_state]))
            target = reward
            if not done:
                target += self.gamma * self.q_a_[next_state, best_a]
            self.q_b_[state, action] += self.alpha * (target - self.q_b_[state, action])

    @property
    def q_table_(self):
        return self.q_a_ + self.q_b_
