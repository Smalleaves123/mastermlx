from __future__ import annotations

import numpy as np
from collections import deque


class DQNAgent:
    """Deep Q-Network with experience replay and a target network."""

    def __init__(self, n_features, n_actions, hidden_sizes=(64, 64), lr=0.001,
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 batch_size=32, memory_size=10000, target_update=100, random_state=None):
        self.n_features = int(n_features)
        self.n_actions = int(n_actions)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.batch_size = int(batch_size)
        self.target_update = int(target_update)
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.memory = deque(maxlen=int(memory_size))
        self._step_count = 0

        # Build network
        sizes = [n_features, *hidden_sizes, n_actions]
        scale = np.sqrt(2.0 / max(n_features, 1))
        self.weights_ = []
        for i in range(len(sizes) - 1):
            W = self._rng.normal(scale=scale, size=(sizes[i], sizes[i + 1]))
            b = np.zeros(sizes[i + 1])
            self.weights_.append((W, b))
        self.target_weights_ = [(W.copy(), b.copy()) for W, b in self.weights_]

    def _forward(self, x, weights):
        h = np.asarray(x, dtype=float).ravel()
        for i, (W, b) in enumerate(weights):
            h = h @ W + b
            if i < len(weights) - 1:
                h = np.maximum(0, h)
        return h

    def select_action(self, state):
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        q = self._forward(state, self.weights_)
        return int(np.argmax(q))

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        self._step_count += 1
        batch_idx = self._rng.choice(len(self.memory), size=self.batch_size, replace=False)
        S, A, R, NS, D = [], [], [], [], []
        for i in batch_idx:
            s, a, r, ns, d = self.memory[i]
            S.append(np.asarray(s, dtype=float).ravel())
            A.append(a)
            R.append(r)
            NS.append(np.asarray(ns, dtype=float).ravel())
            D.append(d)
        S = np.array(S)
        NS = np.array(NS)

        # Forward pass — current Q
        activations = [S]
        h = S
        for W, b in self.weights_[:-1]:
            h = h @ W + b
            h = np.maximum(0, h)
            activations.append(h)
        Q = h @ self.weights_[-1][0] + self.weights_[-1][1]

        # Target Q
        h = NS
        for W, b in self.target_weights_[:-1]:
            h = h @ W + b
            h = np.maximum(0, h)
        Q_next = h @ self.target_weights_[-1][0] + self.target_weights_[-1][1]
        max_next = np.max(Q_next, axis=1)

        # Build target: only update the chosen action's Q value
        target = Q.copy()
        for i in range(self.batch_size):
            target[i, A[i]] = R[i]
            if not D[i]:
                target[i, A[i]] += self.gamma * max_next[i]

        # Backward: MSE loss gradients
        diff = (Q - target) / self.batch_size
        dW = activations[-1].T @ diff
        db = np.sum(diff, axis=0)
        self.weights_[-1] = (self.weights_[-1][0] - self.lr * dW,
                              self.weights_[-1][1] - self.lr * db)
        dout = diff @ self.weights_[-1][0].T
        for i in range(len(self.weights_) - 2, -1, -1):
            dout = dout * (activations[i + 1] > 0)
            dW = activations[i].T @ dout
            db = np.sum(dout, axis=0)
            self.weights_[i] = (self.weights_[i][0] - self.lr * dW,
                                 self.weights_[i][1] - self.lr * db)
            if i > 0:
                dout = dout @ self.weights_[i][0].T

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if self._step_count % self.target_update == 0:
            self.target_weights_ = [(W.copy(), b.copy()) for W, b in self.weights_]
