from __future__ import annotations

import numpy as np

from ..config import get_backend

from .lqr import finite_horizon_lqr

try:
    from ._control_ops import (
        finite_difference_jacobian as _cy_finite_difference_jacobian,
        quadratic_trajectory_cost as _cy_quadratic_trajectory_cost,
        rollout_dynamics as _cy_rollout_dynamics,
    )
except ImportError:  # pragma: no cover - fallback when Cython extensions are unavailable
    _cy_finite_difference_jacobian = None
    _cy_quadratic_trajectory_cost = None
    _cy_rollout_dynamics = None


def rollout_dynamics(f, x0, U, dt=None, args=None):
    """Roll out a discrete or continuous-time system."""

    if get_backend() != "numpy" and _cy_rollout_dynamics is not None:
        return _cy_rollout_dynamics(f, x0, U, dt=dt, args=args)

    x = np.asarray(x0, dtype=float).reshape(-1)
    states = [x.copy()]
    args = () if args is None else tuple(args)
    for u in U:
        u = np.asarray(u, dtype=float).reshape(-1)
        if dt is None:
            x = np.asarray(f(x, u, *args), dtype=float).reshape(-1)
        else:
            dx = np.asarray(f(x, u, *args), dtype=float).reshape(-1)
            x = x + float(dt) * dx
        states.append(x.copy())
    return np.asarray(states)


class LinearMPC:
    """Unconstrained linear MPC solved with finite-horizon LQR."""

    def __init__(self, A, B, Q, R, horizon, Qf=None, u_bounds=None):
        self.A_ = np.asarray(A, dtype=float)
        self.B_ = np.asarray(B, dtype=float)
        self.Q_ = np.asarray(Q, dtype=float)
        self.R_ = np.asarray(R, dtype=float)
        self.horizon = int(horizon)
        self.Qf_ = None if Qf is None else np.asarray(Qf, dtype=float)
        self.u_bounds_ = u_bounds
        self.K_, self.P_, self.reference_ = finite_horizon_lqr(self.A_, self.B_, self.Q_, self.R_, self.horizon, self.Qf_)

    def _clip(self, u):
        if self.u_bounds_ is None:
            return u
        lo, hi = self.u_bounds_
        if lo is not None:
            u = np.maximum(u, lo)
        if hi is not None:
            u = np.minimum(u, hi)
        return u

    def control(self, x, x_ref=None, t=0):
        if not self.K_:
            raise RuntimeError("controller has no gains")
        idx = min(int(t), len(self.K_) - 1)
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        if x_ref is None:
            x_ref = np.zeros_like(x)
        else:
            x_ref = np.asarray(x_ref, dtype=float).reshape(-1, 1)
        u = -self.K_[idx] @ (x - x_ref)
        return self._clip(u.ravel())


def _finite_difference_jacobian(f, x, u, eps=1e-5):
    if get_backend() != "numpy" and _cy_finite_difference_jacobian is not None:
        return _cy_finite_difference_jacobian(f, np.asarray(x, dtype=float).reshape(-1), np.asarray(u, dtype=float).reshape(-1), eps=eps)

    x = np.asarray(x, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)
    fx = np.asarray(f(x, u), dtype=float).reshape(-1)
    n = x.size
    m = u.size
    A = np.zeros((fx.size, n), dtype=float)
    B = np.zeros((fx.size, m), dtype=float)
    for i in range(n):
        dx = np.zeros_like(x)
        dx[i] = eps
        A[:, i] = (np.asarray(f(x + dx, u), dtype=float).reshape(-1) - np.asarray(f(x - dx, u), dtype=float).reshape(-1)) / (2.0 * eps)
    for i in range(m):
        du = np.zeros_like(u)
        du[i] = eps
        B[:, i] = (np.asarray(f(x, u + du), dtype=float).reshape(-1) - np.asarray(f(x, u - du), dtype=float).reshape(-1)) / (2.0 * eps)
    return A, B


def iLQR(
    dynamics,
    x0,
    U0,
    Q,
    R,
    Qf,
    *,
    x_ref=None,
    u_ref=None,
    max_iter=50,
    tol=1e-6,
    line_search=(1.0, 0.5, 0.25, 0.1, 0.05),
    eps=1e-5,
):
    """Iterative LQR for nonlinear discrete-time dynamics.

    The dynamics function must be of the form `x_next = dynamics(x, u)`.
    Returns the optimized control sequence, state rollout, and cost.
    """

    x0 = np.asarray(x0, dtype=float).reshape(-1)
    U = np.asarray(U0, dtype=float)
    if U.ndim != 2:
        raise ValueError("U0 must have shape (T, control_dim)")
    T, m = U.shape
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    Qf = np.asarray(Qf, dtype=float)

    def rollout(U_seq):
        X = rollout_dynamics(dynamics, x0, U_seq)
        cost = _cy_quadratic_trajectory_cost(X, U_seq, Q, R, Qf, x_ref=x_ref, u_ref=u_ref) if get_backend() != "numpy" and _cy_quadratic_trajectory_cost is not None else None
        if cost is None:
            cost = 0.0
            x_ref_seq = None if x_ref is None else np.asarray(x_ref, dtype=float)
            u_ref_seq = None if u_ref is None else np.asarray(u_ref, dtype=float)
            for t in range(T):
                xr = np.zeros_like(X[t]) if x_ref_seq is None else x_ref_seq[t]
                ur = np.zeros_like(U_seq[t]) if u_ref_seq is None else u_ref_seq[t]
                dx = X[t] - xr
                du = U_seq[t] - ur
                cost += float(dx @ Q @ dx + du @ R @ du)
            xr = np.zeros_like(X[T]) if x_ref_seq is None else x_ref_seq[T]
            dx = X[T] - xr
            cost += float(dx @ Qf @ dx)
        return np.asarray(X), cost

    def stage_cost_terms(x, u, t):
        xr = np.zeros_like(x) if x_ref is None else np.asarray(x_ref, dtype=float)[t]
        ur = np.zeros_like(u) if u_ref is None else np.asarray(u_ref, dtype=float)[t]
        dx = x - xr
        du = u - ur
        return dx, du

    X, cost = rollout(U)
    for _ in range(int(max_iter)):
        prev_cost = cost
        A_seq = []
        B_seq = []
        for t in range(T):
            A_t, B_t = _finite_difference_jacobian(dynamics, X[t], U[t], eps=eps)
            A_seq.append(A_t)
            B_seq.append(B_t)

        V_x = 2.0 * Qf @ (X[T] - (np.zeros_like(X[T]) if x_ref is None else np.asarray(x_ref, dtype=float)[T]))
        V_xx = 2.0 * Qf.copy()
        k_seq = [None] * T
        K_seq = [None] * T

        for t in range(T - 1, -1, -1):
            x = X[t]
            u = U[t]
            dx, du = stage_cost_terms(x, u, t)
            l_x = 2.0 * Q @ dx
            l_u = 2.0 * R @ du
            l_xx = 2.0 * Q
            l_uu = 2.0 * R
            l_ux = np.zeros((m, x.size), dtype=float)

            A_t = A_seq[t]
            B_t = B_seq[t]

            Q_x = l_x + A_t.T @ V_x
            Q_u = l_u + B_t.T @ V_x
            Q_xx = l_xx + A_t.T @ V_xx @ A_t
            Q_ux = l_ux + B_t.T @ V_xx @ A_t
            Q_uu = l_uu + B_t.T @ V_xx @ B_t
            Q_uu = Q_uu + 1e-8 * np.eye(Q_uu.shape[0], dtype=float)

            K_t = np.linalg.solve(Q_uu, Q_ux)
            k_t = np.linalg.solve(Q_uu, Q_u)
            K_seq[t] = K_t
            k_seq[t] = k_t

            V_x = Q_x - K_t.T @ Q_uu @ k_t
            V_xx = Q_xx - K_t.T @ Q_uu @ K_t
            V_xx = 0.5 * (V_xx + V_xx.T)

        improved = False
        best_X, best_cost, best_U = X, cost, U
        for alpha in line_search:
            X_new = [x0.copy()]
            U_new = []
            for t in range(T):
                du = -alpha * k_seq[t] - K_seq[t] @ (X_new[-1] - X[t])
                u_new = U[t] + du
                U_new.append(u_new)
                X_new.append(np.asarray(dynamics(X_new[-1], u_new), dtype=float).reshape(-1))
            X_new = np.asarray(X_new)
            U_new = np.asarray(U_new)
            new_cost = _cy_quadratic_trajectory_cost(X_new, U_new, Q, R, Qf, x_ref=x_ref, u_ref=u_ref) if get_backend() != "numpy" and _cy_quadratic_trajectory_cost is not None else 0.0
            if get_backend() == "numpy" or _cy_quadratic_trajectory_cost is None:
                x_ref_seq = None if x_ref is None else np.asarray(x_ref, dtype=float)
                u_ref_seq = None if u_ref is None else np.asarray(u_ref, dtype=float)
                for t in range(T):
                    xr = np.zeros_like(X_new[t]) if x_ref_seq is None else x_ref_seq[t]
                    ur = np.zeros_like(U_new[t]) if u_ref_seq is None else u_ref_seq[t]
                    dx = X_new[t] - xr
                    du = U_new[t] - ur
                    new_cost += float(dx @ Q @ dx + du @ R @ du)
                xr = np.zeros_like(X_new[T]) if x_ref_seq is None else x_ref_seq[T]
                dx = X_new[T] - xr
                new_cost += float(dx @ Qf @ dx)

            if new_cost < best_cost:
                best_X, best_cost, best_U = X_new, new_cost, U_new
                improved = True
                break

        X, cost, U = best_X, best_cost, best_U
        if not improved or abs(prev_cost - cost) <= tol:
            break

    return U, X, cost
