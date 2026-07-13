"""Compare Python and C++ grid A* paths."""

import time

import numpy as np

from mastermlx.graphs.core import _astar_cpp, _astar_py, astar


grid = np.zeros((300, 300), dtype=int)
for col in range(30, 300, 30):
    grid[100:260, col] = 1
    grid[180, col] = 0
start, goal = (0, 0), (299, 299)


def run(fn, count=3):
    times = []
    for _ in range(count):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


py_t = run(lambda: _astar_py(grid, start, goal, False))
cpp_t = run(lambda: astar(grid, start, goal))
print(f"Python A*: {py_t:.6f}s")
print(f"C++ A*:    {cpp_t:.6f}s")
print(f"C++ loaded: {_astar_cpp is not None}")
print(f"Speedup:    {py_t / cpp_t:.2f}x")
