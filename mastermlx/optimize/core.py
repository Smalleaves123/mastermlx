from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import get_backend

try:
    from ._quad_cpp import quad_gd as _quad_cpp
except ImportError:  # pragma: no cover - fallback when the extension is unavailable
    _quad_cpp = None


@dataclass
class Result:
    """Result returned by a numerical optimizer."""

    x: np.ndarray
    fun: float
    nit: int
    success: bool
    message: str
    history: np.ndarray


def finite_diff(fun, x, eps=1e-6):
    """Estimate a scalar objective gradient with centered differences."""

    if not callable(fun):
        raise TypeError("fun must be callable")
    eps = float(eps)
    if eps <= 0:
        raise ValueError("eps must be positive")
    x = np.asarray(x, dtype=float).reshape(-1)
    grad = np.empty_like(x)
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = eps
        grad[i] = (float(fun(x + step)) - float(fun(x - step))) / (2.0 * eps)
    return grad


def _bounds(bounds, n):
    if bounds is None:
        return None
    if len(bounds) != 2:
        raise ValueError("bounds must be a (lower, upper) pair")
    lo, hi = bounds
    lo = np.full(n, -np.inf) if lo is None else np.broadcast_to(lo, n).astype(float)
    hi = np.full(n, np.inf) if hi is None else np.broadcast_to(hi, n).astype(float)
    if np.any(lo > hi):
        raise ValueError("lower bounds must not exceed upper bounds")
    return lo, hi


def gd(fun, x0, jac=None, lr=1e-2, max_iter=1000, tol=1e-6, bounds=None, callback=None):
    """Minimize a scalar function with gradient descent."""

    if not callable(fun):
        raise TypeError("fun must be callable")
    if jac is not None and not callable(jac):
        raise TypeError("jac must be callable or None")
    lr = float(lr)
    tol = float(tol)
    max_iter = int(max_iter)
    if lr <= 0:
        raise ValueError("lr must be positive")
    if tol < 0:
        raise ValueError("tol must be non-negative")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    x = np.asarray(x0, dtype=float).reshape(-1).copy()
    lim = _bounds(bounds, x.size)
    if lim is not None:
        x = np.clip(x, lim[0], lim[1])

    hist = [float(fun(x))]
    success = False
    message = "maximum iterations reached"
    nit = 0
    for nit in range(1, max_iter + 1):
        g = np.asarray(jac(x) if jac is not None else finite_diff(fun, x), dtype=float).reshape(-1)
        if g.shape != x.shape or not np.all(np.isfinite(g)):
            raise ValueError("jac must return a finite gradient with the same shape as x0")
        if np.linalg.norm(g, ord=np.inf) <= tol:
            success = True
            message = "gradient tolerance reached"
            break

        old = x.copy()
        x = x - lr * g
        if lim is not None:
            x = np.clip(x, lim[0], lim[1])
        value = float(fun(x))
        if not np.isfinite(value):
            raise ValueError("fun returned a non-finite value")
        hist.append(value)
        if callback is not None:
            callback(x.copy(), value, g.copy())
        if np.linalg.norm(x - old, ord=np.inf) <= tol:
            success = True
            message = "step tolerance reached"
            break

    return Result(
        x=x,
        fun=float(fun(x)),
        nit=nit,
        success=success,
        message=message,
        history=np.asarray(hist, dtype=float),
    )


def minimize(fun, x0, method="gd", **kwargs):
    """Minimize a scalar function with a supported numerical method."""

    method = str(method).lower()
    if method in {"gd", "gradient_descent"}:
        return gd(fun, x0, **kwargs)
    raise ValueError("method must be 'gd'")


def quad_gd(H, b, x0, lr=1e-2, max_iter=1000, tol=1e-6, bounds=None):
    """Minimize ``0.5 * x.T @ H @ x + b.T @ x``.

    The C++ backend is used when available and no bounds are requested.
    ``H`` should be symmetric positive semidefinite for stable convergence.
    ``H``, ``b``, and ``x0`` must be finite and non-empty.
    """

    H = np.asarray(H, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be a square matrix")
    if b.size != x0.size or H.shape[0] != x0.size:
        raise ValueError("H, b, and x0 have incompatible shapes")
    if x0.size == 0:
        raise ValueError("H, b, and x0 must be non-empty")
    if not np.all(np.isfinite(H)) or not np.all(np.isfinite(b)) or not np.all(np.isfinite(x0)):
        raise ValueError("H, b, and x0 must contain only finite values")
    lr = float(lr)
    max_iter = int(max_iter)
    tol = float(tol)
    if lr <= 0.0 or max_iter < 1 or tol < 0.0:
        raise ValueError("lr must be positive, max_iter at least 1, and tol non-negative")

    def fun(x):
        return float(0.5 * x @ H @ x + b @ x)

    def jac(x):
        return H @ x + b

    if get_backend() != "numpy" and _quad_cpp is not None and bounds is None:
        x, hist, nit, success = _quad_cpp(H, b, x0, lr, max_iter, tol)
        return Result(
            x=np.asarray(x, dtype=float),
            fun=fun(np.asarray(x, dtype=float)),
            nit=int(nit),
            success=bool(success),
            message="gradient tolerance reached" if success else "maximum iterations reached",
            history=np.asarray(hist, dtype=float)[: int(nit) + 1],
        )
    return gd(fun, x0, jac=jac, lr=lr, max_iter=max_iter, tol=tol, bounds=bounds)


__all__ = ["Result", "finite_diff", "gd", "minimize", "quad_gd"]
