from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

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


def _clear(a, b, hit, step, edge_free=None):
    if edge_free is not None:
        return bool(edge_free(a, b, step))
    if hit is None:
        return True
    dist = float(np.linalg.norm(b - a))
    n = max(1, int(math.ceil(dist / max(step, 1e-12))))
    for t in np.linspace(0.0, 1.0, n + 1):
        if not _free(a + t * (b - a), hit):
            return False
    return True


def _validate_workers(workers):
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    return workers


class _EdgeQueryPool:
    """Run ordered edge checks through a bounded, optional worker pool."""

    def __init__(self, hit, step, edge_free=None, workers=1):
        self.hit = hit
        self.step = step
        self.edge_free = edge_free
        self.executor = None if workers == 1 else ThreadPoolExecutor(max_workers=workers)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
        return False

    def check(self, edges):
        edges = list(edges)
        if not edges:
            return []

        def evaluate(edge):
            return _clear(edge[0], edge[1], self.hit, self.step, edge_free=self.edge_free)

        if self.executor is None:
            return [evaluate(edge) for edge in edges]
        return list(self.executor.map(evaluate, edges))


def _path(nodes, parents, idx):
    out = []
    while idx >= 0:
        out.append(nodes[idx])
        idx = parents[idx]
    return np.asarray(out[::-1], dtype=float)


def _grow_node_storage(nodes, count, maximum):
    capacity = min(maximum, max(count + 1, 2 * nodes.shape[0]))
    expanded = np.empty((capacity, nodes.shape[1]), dtype=float)
    expanded[:count] = nodes[:count]
    return expanded


def _reparent(index, new_parent, new_cost, parents, costs, children):
    """Reparent a tree node and propagate its cost change to descendants."""

    old_parent = parents[index]
    if old_parent >= 0:
        children[old_parent].remove(index)
    parents[index] = new_parent
    children[new_parent].add(index)

    cost_delta = new_cost - costs[index]
    costs[index] = new_cost
    descendants = list(children[index])
    while descendants:
        descendant = descendants.pop()
        costs[descendant] += cost_delta
        descendants.extend(children[descendant])


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
    edge_free=None,
    workers=1,
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
    _validate_workers(workers)
    if not _free(start, hit) or not _free(goal, hit):
        raise ValueError("start and goal must be free")
    if np.array_equal(start, goal):
        return start[None, :]

    rng = np.random.default_rng(random_state)
    maximum_nodes = max_iter + 2
    nodes = np.empty((min(maximum_nodes, 1024), start.size), dtype=float)
    nodes[0] = start
    node_count = 1
    parents = [-1]
    for _ in range(max_iter):
        sample = goal if rng.random() < goal_rate else rng.uniform(bounds[:, 0], bounds[:, 1])
        dist = np.sum((nodes[:node_count] - sample) ** 2, axis=1)
        near_idx = int(np.argmin(dist))
        near = nodes[near_idx]
        delta = sample - near
        length = float(np.linalg.norm(delta))
        if length == 0.0:
            continue
        new = near + delta * min(step, length) / length
        if not _free(new, hit) or not _clear(near, new, hit, collision_step, edge_free=edge_free):
            continue
        if node_count == nodes.shape[0]:
            nodes = _grow_node_storage(nodes, node_count, maximum_nodes)
        nodes[node_count] = new
        parents.append(near_idx)
        node_count += 1
        if np.linalg.norm(new - goal) <= step and _clear(
            new, goal, hit, collision_step, edge_free=edge_free
        ):
            if node_count == nodes.shape[0]:
                nodes = _grow_node_storage(nodes, node_count, maximum_nodes)
            nodes[node_count] = goal
            parents.append(node_count - 1)
            node_count += 1
            return _path(nodes, parents, node_count - 1)
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
    edge_free=None,
    workers=1,
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
    workers = _validate_workers(workers)
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
    maximum_nodes = max_iter + 2
    nodes = np.empty((min(maximum_nodes, 1024), start.size), dtype=float)
    nodes[0] = start
    node_count = 1
    parents = [-1]
    costs = [0.0]
    children: list[set[int]] = [set()]
    best_goal = None
    best_cost = float("inf")

    with _EdgeQueryPool(hit, collision_step, edge_free=edge_free, workers=workers) as queries:
        for _ in range(max_iter):
            sample = goal if rng.random() < goal_rate else rng.uniform(bounds[:, 0], bounds[:, 1])
            distances = np.linalg.norm(nodes[:node_count] - sample, axis=1)
            nearest_index = int(np.argmin(distances))
            nearest = nodes[nearest_index]
            delta = sample - nearest
            length = float(np.linalg.norm(delta))
            if length == 0.0:
                continue
            new = nearest + delta * min(step, length) / length
            if not _free(new, hit) or not queries.check([(nearest, new)])[0]:
                continue

            new_distances = np.linalg.norm(nodes[:node_count] - new, axis=1)
            near_indices = np.flatnonzero(new_distances <= search_radius)
            parent = nearest_index
            parent_cost = costs[nearest_index] + float(np.linalg.norm(new - nearest))
            parent_candidates = []
            for index in near_indices:
                index = int(index)
                candidate = nodes[index]
                edge_cost = float(new_distances[index])
                cost = costs[index] + edge_cost
                if cost < parent_cost:
                    parent_candidates.append((index, cost, candidate))
            parent_checks = queries.check((candidate, new) for _, _, candidate in parent_candidates)
            for (index, cost, _), is_free in zip(parent_candidates, parent_checks):
                if is_free and cost < parent_cost:
                    parent = index
                    parent_cost = cost

            if node_count == nodes.shape[0]:
                nodes = _grow_node_storage(nodes, node_count, maximum_nodes)
            nodes[node_count] = new
            parents.append(parent)
            costs.append(parent_cost)
            children.append(set())
            new_index = node_count
            children[parent].add(new_index)
            node_count += 1

            rewire_candidates = []
            for index in near_indices:
                index = int(index)
                if index == parent:
                    continue
                edge_cost = float(new_distances[index])
                rewired_cost = parent_cost + edge_cost
                if rewired_cost + 1e-12 < costs[index]:
                    rewire_candidates.append((index, rewired_cost, nodes[index]))
            rewire_checks = queries.check((new, candidate) for _, _, candidate in rewire_candidates)
            for (index, rewired_cost, _), is_free in zip(rewire_candidates, rewire_checks):
                if is_free and rewired_cost + 1e-12 < costs[index]:
                    _reparent(index, new_index, rewired_cost, parents, costs, children)

            if best_goal is not None:
                best_cost = costs[best_goal]

            distance_to_goal = float(np.linalg.norm(new - goal))
            total_goal_cost = parent_cost + distance_to_goal
            if (
                distance_to_goal <= goal_tolerance
                and total_goal_cost < best_cost
                and queries.check([(new, goal)])[0]
            ):
                if best_goal is None:
                    if node_count == nodes.shape[0]:
                        nodes = _grow_node_storage(nodes, node_count, maximum_nodes)
                    nodes[node_count] = goal
                    parents.append(new_index)
                    costs.append(total_goal_cost)
                    children.append(set())
                    best_goal = node_count
                    children[new_index].add(best_goal)
                    node_count += 1
                else:
                    _reparent(best_goal, new_index, total_goal_cost, parents, costs, children)
                best_cost = total_goal_cost
                if stop_on_first_path:
                    return _path(nodes, parents, best_goal)

    return None if best_goal is None else _path(nodes, parents, best_goal)


def smooth(path, hit=None, n=100, random_state=None, edge_free=None, workers=1):
    """Shortcut a path in an arbitrary-dimensional state space."""

    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] < 1 or path.shape[0] < 2:
        raise ValueError("path must have shape (n, n_dims) with at least two points")
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    _validate_workers(workers)
    rng = np.random.default_rng(random_state)
    out = path.copy()
    for _ in range(n):
        if out.shape[0] <= 2:
            break
        i, j = sorted(rng.integers(0, out.shape[0], size=2))
        if j <= i + 1 or not _clear(out[i], out[j], hit, 0.01, edge_free=edge_free):
            continue
        out = np.concatenate([out[: i + 1], out[j:]], axis=0)
    return out


__all__ = ["rrt", "rrt_star", "smooth"]
