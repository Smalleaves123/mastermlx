from __future__ import annotations

from collections import deque
import heapq
import math
import numbers

import numpy as np

from ..config import get_backend

try:
    from ._grid_cpp import astar as _astar_cpp
except ImportError:  # pragma: no cover - fallback when the extension is unavailable
    _astar_cpp = None


def _next(graph, node):
    if callable(graph):
        return graph(node)
    if hasattr(graph, "get"):
        return graph.get(node, ())
    raise TypeError("graph must be a mapping or callable")


def _path(prev, start, goal):
    if goal not in prev:
        return None
    out = []
    node = goal
    while node is not None:
        out.append(node)
        node = prev[node]
    out.reverse()
    return out


def bfs(graph, start, goal=None):
    """Breadth-first traversal or shortest unweighted path."""

    q = deque([start])
    prev = {start: None}
    order = []
    while q:
        node = q.popleft()
        order.append(node)
        if node == goal:
            break
        for nxt in _next(graph, node):
            if nxt not in prev:
                prev[nxt] = node
                q.append(nxt)
    return order if goal is None else _path(prev, start, goal)


def dfs(graph, start, goal=None):
    """Depth-first traversal or path to a goal."""

    stack = [start]
    prev = {start: None}
    order = []
    while stack:
        node = stack.pop()
        order.append(node)
        if node == goal:
            break
        nxts = list(_next(graph, node))
        for nxt in reversed(nxts):
            if nxt not in prev:
                prev[nxt] = node
                stack.append(nxt)
    return order if goal is None else _path(prev, start, goal)


def _edges(graph, node):
    for item in _next(graph, node):
        if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], numbers.Real):
            nxt, weight = item
        else:
            nxt, weight = item, 1.0
        weight = float(weight)
        if weight < 0:
            raise ValueError("dijkstra does not support negative weights")
        yield nxt, weight


def dijkstra(graph, start, goal=None):
    """Find weighted shortest paths.

    With ``goal`` set, return ``(path, cost)``. Otherwise return all distances.
    Weighted neighbors use ``(node, weight)`` pairs.
    """

    dist = {start: 0.0}
    prev = {start: None}
    heap = [(0.0, 0, start)]
    order = 0
    while heap:
        cost, _, node = heapq.heappop(heap)
        if cost > dist[node]:
            continue
        if node == goal:
            break
        for nxt, weight in _edges(graph, node):
            new = cost + weight
            if new < dist.get(nxt, math.inf):
                order += 1
                dist[nxt] = new
                prev[nxt] = node
                heapq.heappush(heap, (new, order, nxt))
    if goal is None:
        return dist
    return _path(prev, start, goal), dist.get(goal, math.inf)


def _grid_check(grid, start, goal):
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError("grid must be a 2D array")
    start = tuple(int(v) for v in start)
    goal = tuple(int(v) for v in goal)
    if len(start) != 2 or len(goal) != 2:
        raise ValueError("start and goal must be 2D coordinates")
    rows, cols = grid.shape
    for point in (start, goal):
        if not (0 <= point[0] < rows and 0 <= point[1] < cols):
            raise ValueError("start and goal must be inside grid")
        if grid[point] != 0:
            raise ValueError("start and goal must be free cells")
    return grid, start, goal


def _h(a, b, diagonal):
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    if diagonal:
        return max(dr, dc) + (math.sqrt(2.0) - 1.0) * min(dr, dc)
    return dr + dc


def _astar_py(grid, start, goal, diagonal):
    rows, cols = grid.shape
    steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
    if diagonal:
        d = math.sqrt(2.0)
        steps += [(-1, -1, d), (-1, 1, d), (1, -1, d), (1, 1, d)]
    heap = [(_h(start, goal, diagonal), 0.0, 0, start)]
    g = {start: 0.0}
    prev = {start: None}
    order = 0
    while heap:
        _, cost, _, node = heapq.heappop(heap)
        if cost > g[node]:
            continue
        if node == goal:
            return _path(prev, start, goal), cost
        for dr, dc, step in steps:
            nxt = (node[0] + dr, node[1] + dc)
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols) or grid[nxt] != 0:
                continue
            if dr and dc and (grid[node[0] + dr, node[1]] != 0 or grid[node[0], node[1] + dc] != 0):
                continue
            new = cost + step
            if new < g.get(nxt, math.inf):
                order += 1
                g[nxt] = new
                prev[nxt] = node
                heapq.heappush(heap, (new + _h(nxt, goal, diagonal), new, order, nxt))
    return None, math.inf


def astar(grid, start, goal, diagonal=False):
    """Find a shortest path through a 2D grid.

    Zero-valued cells are free and nonzero cells are blocked. Returns
    ``(path, cost)`` or ``(None, inf)`` when no path exists.
    """

    grid, start, goal = _grid_check(grid, start, goal)
    if get_backend() != "numpy" and _astar_cpp is not None:
        path, cost = _astar_cpp(np.asarray(grid, dtype=np.int32), start, goal, bool(diagonal))
        if path is None:
            return None, math.inf
        return [tuple(int(v) for v in point) for point in path], float(cost)
    return _astar_py(grid, start, goal, bool(diagonal))


__all__ = ["astar", "bfs", "dfs", "dijkstra"]
