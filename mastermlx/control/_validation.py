"""Validation helpers shared by control algorithms."""

from __future__ import annotations

import numpy as np


def validate_lqr_matrices(A, B, Q, R):
    """Validate and normalize matrices for discrete-time LQR/MPC."""

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D matrix")
    n = A.shape[0]
    if B.ndim != 2 or B.shape[0] != n or B.shape[1] < 1:
        raise ValueError("B must have shape (A.shape[0], control_dim)")
    m = B.shape[1]
    if Q.shape != (n, n):
        raise ValueError("Q must have the same square shape as A")
    if R.shape != (m, m):
        raise ValueError("R must be square with size B.shape[1]")
    for name, matrix in (("A", A), ("B", B), ("Q", Q), ("R", R)):
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(Q, Q.T, atol=1e-10, rtol=1e-10):
        raise ValueError("Q must be symmetric")
    if not np.allclose(R, R.T, atol=1e-10, rtol=1e-10):
        raise ValueError("R must be symmetric")
    if np.min(np.linalg.eigvalsh(R)) <= 0.0:
        raise ValueError("R must be positive definite")
    return A, B, Q, R


def validate_iteration_options(max_iter, tol):
    max_iter = int(max_iter)
    tol = float(tol)
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be positive and finite")
    return max_iter, tol
