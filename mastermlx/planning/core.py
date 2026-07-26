from __future__ import annotations

import math

import numpy as np


def _bounds(bounds, dims=None):
    out = np.asarray(bounds, dtype=float)
    if out.ndim != 2 or out.shape[1] != 2 or out.shape[0] < 1 or np.any(out[:, 0] >= out[:, 1]):
        raise ValueError("bounds must have shape (n_dims, 2) with lower < upper")
    if dims is not None and out.shape[0] != dims:
        raise ValueError(f"bounds must have {dims} rows")
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


def rrt(
    start,
    goal,
    bounds,
    hit=None,
    step=0.1,
    goal_rate=0.1,
    max_iter=5000,
    random_state=None,
    collision_step=None,
):
    """Plan a collision-free path in an arbitrary-dimensional state space."""

    start = np.asarray(start, dtype=float).reshape(-1)
    goal = np.asarray(goal, dtype=float).reshape(-1)
    if start.size == 0 or goal.size != start.size:
        raise ValueError("start and goal must have the same non-zero dimension")
    bounds = _bounds(bounds, start.size)
    if np.any(start < bounds[:, 0]) or np.any(start > bounds[:, 1]):
        raise ValueError("start must be inside bounds")
    if np.any(goal < bounds[:, 0]) or np.any(goal > bounds[:, 1]):
        raise ValueError("goal must be inside bounds")
    step = float(step)
    goal_rate = float(goal_rate)
    max_iter = int(max_iter)
    if step <= 0 or not 0 <= goal_rate <= 1 or max_iter < 1:
        raise ValueError("step must be positive, goal_rate in [0, 1], max_iter at least 1")
    collision_step = step * 0.5 if collision_step is None else float(collision_step)
    if collision_step <= 0.0 or not np.isfinite(collision_step):
        raise ValueError("collision_step must be a positive finite value")
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
        if not _free(new, hit) or not _clear(near, new, hit, collision_step):
            continue
        nodes.append(new)
        parents.append(len(nodes) - 2)
        if np.linalg.norm(new - goal) <= step and _clear(new, goal, hit, collision_step):
            nodes.append(goal.copy())
            parents.append(len(nodes) - 2)
            return _path(nodes, parents, len(nodes) - 1)
    return None


def rrt_star(
    start,
    goal,
    bounds,
    hit=None,
    step=0.1,
    goal_rate=0.1,
    max_iter=5000,
    search_radius=None,
    goal_tolerance=None,
    random_state=None,
    collision_step=None,
    stop_on_first_path=False,
):
    """Plan a collision-free path with RRT* rewiring.

    The planner keeps the same lightweight contract as :func:`rrt`, while
    searching for lower-cost parents and rewiring nearby nodes when a shorter
    collision-free connection is found.  ``None`` is returned when no path is
    found within ``max_iter``.
    """

    start = np.asarray(start, dtype=float).reshape(-1)
    goal = np.asarray(goal, dtype=float).reshape(-1)
    if start.size == 0 or goal.size != start.size:
        raise ValueError("start and goal must have the same non-zero dimension")
    bounds = _bounds(bounds, start.size)
    if np.any(start < bounds[:, 0]) or np.any(start > bounds[:, 1]):
        raise ValueError("start must be inside bounds")
    if np.any(goal < bounds[:, 0]) or np.any(goal > bounds[:, 1]):
        raise ValueError("goal must be inside bounds")
    step = float(step)
    goal_rate = float(goal_rate)
    max_iter = int(max_iter)
    if step <= 0 or not 0 <= goal_rate <= 1 or max_iter < 1:
        raise ValueError("step must be positive, goal_rate in [0, 1], max_iter at least 1")
    collision_step = step * 0.5 if collision_step is None else float(collision_step)
    if collision_step <= 0.0 or not np.isfinite(collision_step):
        raise ValueError("collision_step must be a positive finite value")
    if search_radius is None:
        search_radius = 4.0 * step
    search_radius = float(search_radius)
    if search_radius <= 0.0 or not np.isfinite(search_radius):
        raise ValueError("search_radius must be a positive finite value")
    if goal_tolerance is None:
        goal_tolerance = step
    goal_tolerance = float(goal_tolerance)
    if goal_tolerance <= 0.0 or not np.isfinite(goal_tolerance):
        raise ValueError("goal_tolerance must be a positive finite value")
    stop_on_first_path = bool(stop_on_first_path)
    if not _free(start, hit) or not _free(goal, hit):
        raise ValueError("start and goal must be free")
    if np.array_equal(start, goal):
        return start[None, :]

    rng = np.random.default_rng(random_state)
    nodes = [start.copy()]
    parents = [-1]
    costs = [0.0]
    best_goal = None
    best_cost = float("inf")

    for _ in range(max_iter):
        sample = goal if rng.random() < goal_rate else rng.uniform(bounds[:, 0], bounds[:, 1])
        distances = np.asarray([np.linalg.norm(node - sample) for node in nodes], dtype=float)
        nearest_index = int(np.argmin(distances))
        nearest = nodes[nearest_index]
        delta = sample - nearest
        length = float(np.linalg.norm(delta))
        if length == 0.0:
            continue
        new = nearest + delta * min(step, length) / length
        if not _free(new, hit) or not _clear(nearest, new, hit, collision_step):
            continue

        near_indices = np.flatnonzero(
            np.asarray([np.linalg.norm(node - new) for node in nodes], dtype=float) <= search_radius
        )
        parent = nearest_index
        parent_cost = costs[nearest_index] + float(np.linalg.norm(new - nearest))
        for index in near_indices:
            candidate = nodes[int(index)]
            edge_cost = float(np.linalg.norm(new - candidate))
            cost = costs[int(index)] + edge_cost
            if cost < parent_cost and _clear(candidate, new, hit, collision_step):
                parent = int(index)
                parent_cost = cost

        nodes.append(new)
        parents.append(parent)
        costs.append(parent_cost)
        new_index = len(nodes) - 1

        for index in near_indices:
            index = int(index)
            if index == parent:
                continue
            edge_cost = float(np.linalg.norm(nodes[index] - new))
            rewired_cost = parent_cost + edge_cost
            if rewired_cost + 1e-12 < costs[index] and _clear(new, nodes[index], hit, collision_step):
                parents[index] = new_index
                costs[index] = rewired_cost

        distance_to_goal = float(np.linalg.norm(new - goal))
        total_goal_cost = parent_cost + distance_to_goal
        if (
            distance_to_goal <= goal_tolerance
            and total_goal_cost < best_cost
            and _clear(new, goal, hit, collision_step)
        ):
            if best_goal is None:
                nodes.append(goal.copy())
                parents.append(new_index)
                costs.append(total_goal_cost)
                best_goal = len(nodes) - 1
            else:
                parents[best_goal] = new_index
                costs[best_goal] = total_goal_cost
            best_cost = total_goal_cost
            if stop_on_first_path:
                return _path(nodes, parents, best_goal)

    return None if best_goal is None else _path(nodes, parents, best_goal)


def smooth(path, hit=None, n=100, random_state=None):
    """Shortcut a path in an arbitrary-dimensional state space."""

    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] < 1 or path.shape[0] < 2:
        raise ValueError("path must have shape (n, n_dims) with at least two points")
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


__all__ = ["rrt", "rrt_star", "smooth"]
