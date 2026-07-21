from __future__ import annotations

from functools import lru_cache
import importlib

import numpy as np
from typing import cast

from ..config import get_backend
from ._validation import validate_iteration_options, validate_lqr_matrices

from .lqr import finite_horizon_lqr


@lru_cache(maxsize=2)
def _load_cpp_control(backend=None):
    if backend is None:
        backend = get_backend()
    if backend != "auto":
        return None
    try:
        return importlib.import_module("mastermlx.control._control_cpp")
    except ImportError:
        return None


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


def rollout_linear_dynamics(A, B, x0, U):
    """Roll out ``x[t+1] = A @ x[t] + B @ U[t]`` without callbacks."""

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D matrix")
    if B.ndim != 2 or B.shape[0] != A.shape[0] or B.shape[1] < 1:
        raise ValueError("B must have shape (A.shape[0], control_dim)")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(B)):
        raise ValueError("A and B must contain only finite values")
    x = np.asarray(x0, dtype=float).reshape(-1)
    U = np.asarray(U, dtype=float)
    if x.shape != (A.shape[0],) or U.ndim != 2 or U.shape[1] != B.shape[1]:
        raise ValueError("x0 and U have incompatible shapes for A and B")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(U)):
        raise ValueError("x0 and U must contain only finite values")
    cpp = _load_cpp_control(get_backend())
    if cpp is not None:
        return cpp.linear_rollout(A, B, x, U)
    states = np.empty((U.shape[0] + 1, x.size), dtype=float)
    states[0] = x
    for t, u in enumerate(U):
        states[t + 1] = A @ states[t] + B @ u
    return states


def control_backend_report() -> dict[str, str | bool]:
    """Report availability of callback-free control acceleration."""

    cpp = _load_cpp_control(get_backend())
    return {"requested": get_backend(), "cpp_control": cpp is not None}


def _normalize_bounds(bounds, size):
    if bounds is None:
        return None
    if len(bounds) != 2:
        raise ValueError("u_bounds must be a (lower, upper) pair")
    normalized: list[np.ndarray | None] = []
    for bound in bounds:
        if bound is None:
            normalized.append(None)
            continue
        value = np.asarray(bound, dtype=float)
        if value.ndim == 0:
            value = np.full(size, float(value))
        elif value.shape != (size,):
            raise ValueError("each control bound must be scalar or shape (control_dim,)")
        if not np.all(np.isfinite(value)):
            raise ValueError("control bounds must be finite")
        normalized.append(value)
    lower, upper = normalized
    if lower is not None and upper is not None and np.any(lower > upper):
        raise ValueError("lower control bounds must not exceed upper bounds")
    return lower, upper


def _reference_sequence(reference, length, size, name):
    if reference is None:
        return np.zeros((length, size), dtype=float)
    value = np.asarray(reference, dtype=float)
    if value.shape == (size,):
        value = np.broadcast_to(value, (length, size)).copy()
    elif value.shape != (length, size):
        raise ValueError(f"{name} must have shape ({size},) or ({length}, {size})")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _prediction_matrices(A, B, horizon):
    n, m = A.shape[0], B.shape[1]
    Sx = np.zeros((n * (horizon + 1), n), dtype=float)
    Su = np.zeros((n * (horizon + 1), m * horizon), dtype=float)
    powers = [np.eye(n, dtype=float)]
    for _ in range(horizon):
        powers.append(A @ powers[-1])
    for t in range(horizon + 1):
        row = slice(t * n, (t + 1) * n)
        Sx[row] = powers[t]
        for j in range(t):
            col = slice(j * m, (j + 1) * m)
            Su[row, col] = powers[t - 1 - j] @ B
    return Sx, Su


class LinearMPC:
    """Linear MPC with LQR feedback or a box-constrained quadratic solve."""

    def __init__(
        self, A, B, Q, R, horizon, Qf=None, u_bounds=None, *, qp_max_iter=200, qp_tol=1e-8
    ):
        self.A_, self.B_, self.Q_, self.R_ = validate_lqr_matrices(A, B, Q, R)
        self.horizon = int(horizon)
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1")
        self.Qf_ = None if Qf is None else np.asarray(Qf, dtype=float)
        if self.Qf_ is not None and self.Qf_.shape != self.Q_.shape:
            raise ValueError("Qf must have the same shape as Q")
        self.u_bounds_ = _normalize_bounds(u_bounds, self.B_.shape[1])
        self.qp_max_iter, self.qp_tol = validate_iteration_options(qp_max_iter, qp_tol)
        self.K_, self.P_, self.reference_ = finite_horizon_lqr(
            self.A_, self.B_, self.Q_, self.R_, self.horizon, self.Qf_
        )
        self._u_sequence = np.zeros((self.horizon, self.B_.shape[1]), dtype=float)
        self.last_qp_iterations_ = 0
        self.qp_converged_ = True
        self._Sx = None
        self._Su = None
        self._H = None
        self._step = None
        if self.u_bounds_ is not None:
            self._prepare_box_qp()

    def _prepare_box_qp(self):
        self._Sx, self._Su = _prediction_matrices(self.A_, self.B_, self.horizon)
        n = self.A_.shape[0]
        qbar = np.zeros((n * (self.horizon + 1), n * (self.horizon + 1)), dtype=float)
        for t in range(self.horizon):
            row = slice(t * n, (t + 1) * n)
            qbar[row, row] = self.Q_
        qbar[self.horizon * n :, self.horizon * n :] = self.Q_ if self.Qf_ is None else self.Qf_
        rbar = np.kron(np.eye(self.horizon), self.R_)
        self._H = self._Su.T @ qbar @ self._Su + rbar
        self._H = 0.5 * (self._H + self._H.T)
        lipschitz = float(np.max(np.linalg.eigvalsh(self._H)))
        if not np.isfinite(lipschitz) or lipschitz <= 0.0:
            raise ValueError("MPC quadratic objective must be positive definite")
        self._step = 1.0 / lipschitz
        self._qbar = qbar
        self._rbar = rbar

    def _clip(self, u):
        if self.u_bounds_ is None:
            return u
        lo, hi = self.u_bounds_
        if lo is not None:
            u = np.maximum(u, lo)
        if hi is not None:
            u = np.minimum(u, hi)
        return u

    def reset(self):
        self._u_sequence.fill(0.0)
        self.last_qp_iterations_ = 0
        self.qp_converged_ = True

    def _solve_box_qp(self, x, x_ref, u_ref):
        n, m = self.A_.shape[0], self.B_.shape[1]
        x_ref_seq = _reference_sequence(x_ref, self.horizon + 1, n, "x_ref")
        u_ref_seq = _reference_sequence(u_ref, self.horizon, m, "u_ref")
        Sx = cast(np.ndarray, self._Sx)
        Su = cast(np.ndarray, self._Su)
        H = cast(np.ndarray, self._H)
        step = cast(float, self._step)
        base = Sx @ x
        q = Su.T @ self._qbar @ (base - x_ref_seq.reshape(-1)) - self._rbar @ u_ref_seq.reshape(-1)
        lower, upper = self.u_bounds_
        lower = -np.inf if lower is None else lower
        upper = np.inf if upper is None else upper
        sequence = self._u_sequence.reshape(-1).copy()
        self.qp_converged_ = False
        for iteration in range(1, self.qp_max_iter + 1):
            updated = np.clip(sequence - step * (H @ sequence + q), lower, upper)
            if np.max(np.abs(updated - sequence)) <= self.qp_tol * (1.0 + np.max(np.abs(sequence))):
                sequence = updated
                self.qp_converged_ = True
                self.last_qp_iterations_ = iteration
                break
            sequence = updated
        else:
            self.last_qp_iterations_ = self.qp_max_iter
        self._u_sequence = sequence.reshape(self.horizon, m)
        control = self._u_sequence[0].copy()
        if self.horizon > 1:
            self._u_sequence[:-1] = self._u_sequence[1:]
            self._u_sequence[-1] = self._u_sequence[-2]
        return control

    def control(self, x, x_ref=None, t=0, u_ref=None):
        if self.K_ is None:
            raise RuntimeError("controller has no gains")
        idx = min(int(t), len(self.K_) - 1)
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape != (self.A_.shape[0],) or not np.all(np.isfinite(x)):
            raise ValueError("x must be a finite state vector with shape (A.shape[0],)")
        if self.u_bounds_ is not None:
            return self._solve_box_qp(x, x_ref, u_ref)
        if x_ref is None:
            ref = np.zeros_like(x)
        else:
            ref_value = np.asarray(x_ref, dtype=float)
            if ref_value.shape == (self.horizon + 1, x.size):
                if not np.all(np.isfinite(ref_value)):
                    raise ValueError("x_ref must contain only finite values")
                ref = ref_value[min(int(t), self.horizon)]
            else:
                ref = _reference_sequence(ref_value, 1, x.size, "x_ref")[0]
        u = -self.K_[idx] @ (x - ref)
        return self._clip(u.ravel())


def _finite_difference_jacobian(f, x, u, eps=1e-5):
    if get_backend() != "numpy" and _cy_finite_difference_jacobian is not None:
        return _cy_finite_difference_jacobian(
            f,
            np.asarray(x, dtype=float).reshape(-1),
            np.asarray(u, dtype=float).reshape(-1),
            eps=eps,
        )

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
        A[:, i] = (
            np.asarray(f(x + dx, u), dtype=float).reshape(-1)
            - np.asarray(f(x - dx, u), dtype=float).reshape(-1)
        ) / (2.0 * eps)
    for i in range(m):
        du = np.zeros_like(u)
        du[i] = eps
        B[:, i] = (
            np.asarray(f(x, u + du), dtype=float).reshape(-1)
            - np.asarray(f(x, u - du), dtype=float).reshape(-1)
        ) / (2.0 * eps)
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
        cost = (
            _cy_quadratic_trajectory_cost(X, U_seq, Q, R, Qf, x_ref=x_ref, u_ref=u_ref)
            if get_backend() != "numpy" and _cy_quadratic_trajectory_cost is not None
            else None
        )
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

        V_x = (
            2.0
            * Qf
            @ (X[T] - (np.zeros_like(X[T]) if x_ref is None else np.asarray(x_ref, dtype=float)[T]))
        )
        V_xx = 2.0 * Qf.copy()
        k_seq: list[np.ndarray | None] = [None] * T
        K_seq: list[np.ndarray | None] = [None] * T

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
            X_new_list = [x0.copy()]
            U_new_list: list[np.ndarray] = []
            for t in range(T):
                du = -alpha * cast(np.ndarray, k_seq[t]) - cast(np.ndarray, K_seq[t]) @ (
                    X_new_list[-1] - X[t]
                )
                u_new = U[t] + du
                U_new_list.append(u_new)
                X_new_list.append(
                    np.asarray(dynamics(X_new_list[-1], u_new), dtype=float).reshape(-1)
                )
            X_new = np.asarray(X_new_list)
            U_new = np.asarray(U_new_list)
            new_cost = (
                _cy_quadratic_trajectory_cost(X_new, U_new, Q, R, Qf, x_ref=x_ref, u_ref=u_ref)
                if get_backend() != "numpy" and _cy_quadratic_trajectory_cost is not None
                else 0.0
            )
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
