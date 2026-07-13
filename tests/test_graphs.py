import numpy as np

from mastermlx import astar, bfs, dfs, dijkstra
from mastermlx.graphs.core import _astar_py


def test_bfs_and_dfs_find_paths():
    graph = {
        "a": ["b", "c"],
        "b": ["d"],
        "c": ["d"],
        "d": [],
    }

    assert bfs(graph, "a", "d") == ["a", "b", "d"]
    assert dfs(graph, "a", "d") == ["a", "b", "d"]


def test_dijkstra_finds_weighted_shortest_path():
    graph = {
        "a": [("b", 2.0), ("c", 5.0)],
        "b": [("c", 1.0), ("d", 5.0)],
        "c": [("d", 1.0)],
        "d": [],
    }

    path, cost = dijkstra(graph, "a", "d")

    assert path == ["a", "b", "c", "d"]
    assert np.isclose(cost, 4.0)


def test_astar_finds_grid_path_and_respects_obstacles():
    grid = np.zeros((6, 6), dtype=int)
    grid[2, :4] = 1

    path, cost = astar(grid, (0, 0), (5, 5))

    assert path[0] == (0, 0)
    assert path[-1] == (5, 5)
    assert all(grid[p] == 0 for p in path)
    assert np.isclose(cost, len(path) - 1)


def test_astar_returns_no_path_when_grid_is_blocked():
    grid = np.zeros((3, 3), dtype=int)
    grid[1, :] = 1

    path, cost = astar(grid, (0, 0), (2, 2))

    assert path is None
    assert np.isinf(cost)


def test_astar_cpp_and_python_paths_have_same_cost():
    grid = np.zeros((8, 8), dtype=int)
    grid[3, 1:7] = 1

    py_path, py_cost = _astar_py(grid, (0, 0), (7, 7), False)
    path, cost = astar(grid, (0, 0), (7, 7))

    assert path is not None
    assert py_path is not None
    assert np.isclose(cost, py_cost)
