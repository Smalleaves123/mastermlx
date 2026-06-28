from __future__ import annotations

import numpy as np

from ..neural_net.layers import Dense
from ..utils.array import batch_iterator


class REINFORCEAgent:
    """Policy gradient agent with a simple neural network policy."""

    def __init__(self, n_features, n_actions, hidden_sizes=(32,), lr=0.01, gamma=0.99, random_state=None):
        self.n_features = int(n_features)
        self.n_actions = int(n_actions)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

        sizes = [n_features, *hidden_sizes, n_actions]
        self.layers = []
        scale = np.sqrt(2.0 / max(n_features, 1))
        for i in range(len(sizes) - 1):
            W = self._rng.normal(scale=scale, size=(sizes[i], sizes[i + 1]))
            b = np.zeros(sizes[i + 1])
            self.layers.append((W, b))

    def _forward(self, x):
        x = np.asarray(x, dtype=float).ravel()
        h = x
        for i, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if i < len(self.layers) - 1:
                h = np.maximum(0, h)  # ReLU
        # Softmax
        h = h - np.max(h)
        e = np.exp(h)
        return e / np.sum(e)

    def select_action(self, state):
        probs = self._forward(state)
        return int(self._rng.choice(self.n_actions, p=probs))

    def update_episode(self, episode):
        """Update policy from a complete episode.

        episode: list of (state, action, reward) tuples.
        """
        T = len(episode)
        returns = np.zeros(T)
        G = 0.0
        for t in range(T - 1, -1, -1):
            G = episode[t][2] + self.gamma * G
            returns[t] = G
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        for t, (state, action, _) in enumerate(episode):
            x = np.asarray(state, dtype=float).ravel()
            h = x
            activations = [h]
            # Forward
            for i, (W, b) in enumerate(self.layers):
                h = h @ W + b
                if i < len(self.layers) - 1:
                    h = np.maximum(0, h)
                else:
                    h = h - np.max(h)
                    e = np.exp(h)
                    h = e / np.sum(e)
                activations.append(h)
            probs = activations[-1]

            # Backward on log-prob of chosen action
            dout = probs.copy()
            dout[action] -= 1.0
            dout *= returns[t]

            for i in range(len(self.layers) - 1, -1, -1):
                W, b = self.layers[i]
                a_prev = activations[i]
                dW = np.outer(a_prev, dout)
                db = dout
                W -= self.lr * dW
                b -= self.lr * db
                if i > 0:
                    dout = (dout @ W.T) * (a_prev > 0)
