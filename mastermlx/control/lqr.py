from __future__ import annotations

import numpy as np


def solve_discrete_are(A, B, Q, R, *, max_iter=1000, tol=1e-9):
    """Solve the discrete algebraic Riccati equation by fixed-point iteration."""

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    P = Q.copy()
    for _ in range(int(max_iter)):
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

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    n = A.shape[0]
    m = B.shape[1]
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    Qf = Q if Qf is None else np.asarray(Qf, dtype=float)
    ref = None if reference is None else np.asarray(reference, dtype=float)

    P = [None] * (horizon + 1)
    K = [None] * horizon
    P[horizon] = Qf.copy()

    for t in range(horizon - 1, -1, -1):
        S = R + B.T @ P[t + 1] @ B
        K[t] = np.linalg.solve(S, B.T @ P[t + 1] @ A)
        P[t] = Q + A.T @ P[t + 1] @ (A - B @ K[t])
    return K, P, ref


class DiscreteLQR:
    """Infinite-horizon discrete-time LQR controller."""

    def __init__(self, A, B, Q, R, *, max_iter=1000, tol=1e-9):
        self.A_ = np.asarray(A, dtype=float)
        self.B_ = np.asarray(B, dtype=float)
        self.Q_ = np.asarray(Q, dtype=float)
        self.R_ = np.asarray(R, dtype=float)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.P_ = None
        self.K_ = None

    def fit(self):
        self.P_ = solve_discrete_are(self.A_, self.B_, self.Q_, self.R_, max_iter=self.max_iter, tol=self.tol)
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
        u = -self.K_ @ (x - x_ref)
        return u.ravel()
