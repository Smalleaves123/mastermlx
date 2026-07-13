from __future__ import annotations

import math

import numpy as np


def _bounds(bounds):
    out = np.asarray(bounds, dtype=float)
    if out.shape != (2, 2) or np.any(out[:, 0] >= out[:, 1]):
        raise ValueError("bounds must have shape (2, 2) with lower < upper")
    return out


def _free(p, hit):
    return hit is None or not bool(hit(np.asarray(p, dtype=float)))


def _clear(a, b, hit, step):
    dist = float(np.linalg.norm(b - a))
    n = max(1, int(math.ceil(dist / max(step, 1e-12))))
    for t in np.linspace(0.0, 1.0, n + 1):
        if not _free(a + t * (b - a), hit):
            return False
    return True


def _path(nodes, parents, idx):
    out = []
    while idx >= 0:
        out.append(nodes[idx])
        idx = parents[idx]
    return np.asarray(out[::-1], dtype=float)


def rrt(start, goal, bounds, hit=None, step=0.1, goal_rate=0.1, max_iter=5000, random_state=None):
    """Plan a 2D collision-free path with a rapidly exploring tree."""

    start = np.asarray(start, dtype=float).reshape(-1)
    goal = np.asarray(goal, dtype=float).reshape(-1)
    bounds = _bounds(bounds)
    if start.size != 2 or goal.size != 2:
        raise ValueError("start and goal must be 2D points")
    if np.any(start < bounds[:, 0]) or np.any(start > bounds[:, 1]):
        raise ValueError("start must be inside bounds")
    if np.any(goal < bounds[:, 0]) or np.any(goal > bounds[:, 1]):
        raise ValueError("goal must be inside bounds")
    step = float(step)
    goal_rate = float(goal_rate)
    max_iter = int(max_iter)
    if step <= 0 or not 0 <= goal_rate <= 1 or max_iter < 1:
        raise ValueError("step must be positive, goal_rate in [0, 1], max_iter at least 1")
    if not _free(start, hit) or not _free(goal, hit):
        raise ValueError("start and goal must be free")
    if np.array_equal(start, goal):
        return start[None, :]

    rng = np.random.default_rng(random_state)
    nodes = [start.copy()]
    parents = [-1]
    for _ in range(max_iter):
        sample = goal if rng.random() < goal_rate else rng.uniform(bounds[:, 0], bounds[:, 1])
        dist = np.asarray([np.sum((node - sample) ** 2) for node in nodes])
        near = nodes[int(np.argmin(dist))]
        delta = sample - near
        length = float(np.linalg.norm(delta))
        if length == 0.0:
            continue
        new = near + delta * min(step, length) / length
        if not _free(new, hit) or not _clear(near, new, hit, step * 0.5):
            continue
        nodes.append(new)
        parents.append(len(nodes) - 2)
        if np.linalg.norm(new - goal) <= step and _clear(new, goal, hit, step * 0.5):
            nodes.append(goal.copy())
            parents.append(len(nodes) - 2)
            return _path(nodes, parents, len(nodes) - 1)
    return None


def smooth(path, hit=None, n=100, random_state=None):
    """Shortcut a 2D path while keeping it collision-free."""

    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2 or path.shape[0] < 2:
        raise ValueError("path must have shape (n, 2) with at least two points")
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = np.random.default_rng(random_state)
    out = path.copy()
    for _ in range(n):
        if out.shape[0] <= 2:
            break
        i, j = sorted(rng.integers(0, out.shape[0], size=2))
        if j <= i + 1 or not _clear(out[i], out[j], hit, 0.01):
            continue
        out = np.concatenate([out[: i + 1], out[j:]], axis=0)
    return out


__all__ = ["rrt", "smooth"]
