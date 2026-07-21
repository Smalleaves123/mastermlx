from __future__ import annotations

import numpy as np
from typing import cast

from ..config import get_backend
from ._validation import validate_iteration_options, validate_lqr_matrices

try:
    from ._lqr_ops import finite_horizon_lqr as _cy_finite_horizon_lqr
    from ._lqr_ops import solve_discrete_are as _cy_solve_discrete_are
except ImportError:  # pragma: no cover - fallback when Cython extensions are unavailable
    _cy_finite_horizon_lqr = None
    _cy_solve_discrete_are = None


def solve_discrete_are(A, B, Q, R, *, max_iter=1000, tol=1e-9):
    """Solve the discrete algebraic Riccati equation by fixed-point iteration."""

    A, B, Q, R = validate_lqr_matrices(A, B, Q, R)
    max_iter, tol = validate_iteration_options(max_iter, tol)
    if get_backend() != "numpy" and _cy_solve_discrete_are is not None:
        return _cy_solve_discrete_are(A, B, Q, R, max_iter, tol)
    P = Q.copy()
    for _ in range(max_iter):
        BtP = B.T @ P
        S = R + BtP @ B
        K = np.linalg.solve(S, BtP @ A)
        P_next = A.T @ P @ A - A.T @ P @ B @ K + Q
        if np.max(np.abs(P_next - P)) <= tol:
            return P_next
        P = P_next
    return P


def finite_horizon_lqr(A, B, Q, R, horizon, Qf=None, reference=None):
    """Finite-horizon discrete LQR solved by Riccati backward recursion.

    Returns the feedback gains and the cost-to-go matrices.
    """

    A, B, Q, R = validate_lqr_matrices(A, B, Q, R)
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    Qf = Q if Qf is None else np.asarray(Qf, dtype=float)
    if Qf.shape != Q.shape or not np.all(np.isfinite(Qf)):
        raise ValueError("Qf must be finite and have the same shape as Q")
    if not np.allclose(Qf, Qf.T, atol=1e-10, rtol=1e-10):
        raise ValueError("Qf must be symmetric")
    ref = None if reference is None else np.asarray(reference, dtype=float)
    if ref is not None and (ref.shape[-1:] != (A.shape[0],) or not np.all(np.isfinite(ref))):
        raise ValueError("reference must contain finite state vectors")

    if get_backend() != "numpy" and _cy_finite_horizon_lqr is not None:
        return _cy_finite_horizon_lqr(A, B, Q, R, horizon, Qf=Qf, reference=ref)

    P: list[np.ndarray | None] = [None] * (horizon + 1)
    K: list[np.ndarray | None] = [None] * horizon
    P[horizon] = Qf.copy()

    for t in range(horizon - 1, -1, -1):
        p_next = cast(np.ndarray, P[t + 1])
        S = R + B.T @ p_next @ B
        gain = np.linalg.solve(S, B.T @ p_next @ A)
        K[t] = gain
        P[t] = Q + A.T @ p_next @ (A - B @ gain)
    return K, P, ref


class DiscreteLQR:
    """Infinite-horizon discrete-time LQR controller."""

    def __init__(self, A, B, Q, R, *, max_iter=1000, tol=1e-9):
        self.max_iter, self.tol = validate_iteration_options(max_iter, tol)
        self.A_, self.B_, self.Q_, self.R_ = validate_lqr_matrices(A, B, Q, R)
        self.P_ = None
        self.K_ = None

    def fit(self):
        self.P_ = solve_discrete_are(
            self.A_, self.B_, self.Q_, self.R_, max_iter=self.max_iter, tol=self.tol
        )
        S = self.R_ + self.B_.T @ self.P_ @ self.B_
        self.K_ = np.linalg.solve(S, self.B_.T @ self.P_ @ self.A_)
        return self

    def control(self, x, x_ref=None):
        if self.K_ is None:
            self.fit()
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        if x_ref is None:
            x_ref = np.zeros_like(x)
        else:
            x_ref = np.asarray(x_ref, dtype=float).reshape(-1, 1)
        u = -cast(np.ndarray, self.K_) @ (x - x_ref)
        return u.ravel()
