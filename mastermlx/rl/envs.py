from __future__ import annotations

import numpy as np


class GridWorld:
    """Simple grid world with a reward at the goal.

    States are flattened indices: state = row * n_cols + col.
    Actions: 0=up, 1=right, 2=down, 3=left.
    """

    def __init__(self, rows=4, cols=4, start=(0, 0), goal=None, walls=None, random_state=None):
        self.rows = int(rows)
        self.cols = int(cols)
        self.start = tuple(start)
        self.goal = tuple(goal) if goal else (rows - 1, cols - 1)
        self.walls = set(walls) if walls else set()
        self._rng = np.random.default_rng(random_state)
        self._pos = self.start

    @property
    def n_states(self):
        return self.rows * self.cols

    @property
    def n_actions(self):
        return 4

    def reset(self):
        self._pos = self.start
        return self._idx(self._pos)

    def step(self, action):
        r, c = self._pos
        if action == 0:     r = max(0, r - 1)
        elif action == 1:   c = min(self.cols - 1, c + 1)
        elif action == 2:   r = min(self.rows - 1, r + 1)
        elif action == 3:   c = max(0, c - 1)
        if (r, c) in self.walls:
            r, c = self._pos
        self._pos = (r, c)
        done = self._pos == self.goal
        return self._idx(self._pos), 1.0 if done else -0.01, done

    def _idx(self, pos):
        return pos[0] * self.cols + pos[1]
