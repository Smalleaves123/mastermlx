"""Compare Python and C++ quadratic optimization paths."""

import time

import numpy as np

from mastermlx.optimize.core import _quad_cpp, gd, quad_gd


rng = np.random.default_rng(0)
n = 120
H = np.diag(np.linspace(1.0, 3.0, n))
b = rng.normal(size=n)
x0 = np.zeros(n)


def fun(x):
    return float(0.5 * x @ H @ x + b @ x)


def jac(x):
    return H @ x + b


def run(fn, count=5):
    times = []
    for _ in range(count):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


py_t = run(lambda: gd(fun, x0, lr=0.3, max_iter=300))
cpp_t = run(lambda: quad_gd(H, b, x0, lr=0.3, max_iter=300))
print(f"Python finite-diff path: {py_t:.6f}s")
print(f"C++ quadratic path:      {cpp_t:.6f}s")
print(f"C++ loaded:              {_quad_cpp is not None}")
print(f"Speedup:              {py_t / cpp_t:.2f}x")
