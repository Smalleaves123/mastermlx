from __future__ import annotations

import numpy as np

from ..config import get_backend

try:
    from ._kalman_ops import kalman_predict as _cy_kalman_predict
    from ._kalman_ops import kalman_update as _cy_kalman_update
except ImportError:  # pragma: no cover - fallback when Cython extensions are unavailable
    _cy_kalman_predict = None
    _cy_kalman_update = None


def _as_2d_matrix(M, name):
    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    return M


class KalmanFilter:
    """Linear Kalman filter for discrete-time systems.

    State model:
        x_k = F x_{k-1} + B u_k + w_k
        z_k = H x_k + v_k
    """

    def __init__(self, x0, P0, F, H, Q, R, B=None):
        self.x_ = np.asarray(x0, dtype=float).reshape(-1, 1)
        self.P_ = _as_2d_matrix(P0, "P0")
        self.F_ = _as_2d_matrix(F, "F")
        self.H_ = _as_2d_matrix(H, "H")
        self.Q_ = _as_2d_matrix(Q, "Q")
        self.R_ = _as_2d_matrix(R, "R")
        self.B_ = None if B is None else _as_2d_matrix(B, "B")
        self.x_prior_ = self.x_.copy()
        self.P_prior_ = self.P_.copy()
        self.x_post_ = self.x_.copy()
        self.P_post_ = self.P_.copy()

    @property
    def state(self):
        return self.x_.ravel()

    def predict(self, u=None, F=None, Q=None, B=None):
        F = self.F_ if F is None else _as_2d_matrix(F, "F")
        Q = self.Q_ if Q is None else _as_2d_matrix(Q, "Q")
        B = self.B_ if B is None else (_as_2d_matrix(B, "B") if B is not None else None)

        if get_backend() != "numpy" and _cy_kalman_predict is not None:
            self.x_, self.P_ = _cy_kalman_predict(self.x_, self.P_, F, Q, B=B, u=u)
            self.x_ = np.asarray(self.x_, dtype=float).reshape(-1, 1)
            self.P_ = np.asarray(self.P_, dtype=float)
            self.x_prior_ = self.x_.copy()
            self.P_prior_ = self.P_.copy()
            return self.state, self.P_.copy()

        x = F @ self.x_
        if u is not None:
            if B is None:
                raise ValueError("Control input provided but no control matrix B is available")
            u = np.asarray(u, dtype=float).reshape(-1, 1)
            x = x + B @ u

        P = F @ self.P_ @ F.T + Q
        self.x_prior_ = x
        self.P_prior_ = P
        self.x_ = x
        self.P_ = P
        return self.state, self.P_.copy()

    def update(self, z, H=None, R=None):
        H = self.H_ if H is None else _as_2d_matrix(H, "H")
        R = self.R_ if R is None else _as_2d_matrix(R, "R")
        if get_backend() != "numpy" and _cy_kalman_update is not None:
            self.x_, self.P_ = _cy_kalman_update(self.x_, self.P_, z, H, R)
            self.x_ = np.asarray(self.x_, dtype=float).reshape(-1, 1)
            self.P_ = np.asarray(self.P_, dtype=float)
            self.x_post_ = self.x_.copy()
            self.P_post_ = self.P_.copy()
            return self.state, self.P_.copy()

        z = np.asarray(z, dtype=float).reshape(-1, 1)

        y = z - H @ self.x_
        S = H @ self.P_ @ H.T + R
        K = self.P_ @ H.T @ np.linalg.solve(S, np.eye(S.shape[0], dtype=float))
        self.x_ = self.x_ + K @ y
        identity = np.eye(self.P_.shape[0], dtype=float)
        self.P_ = (identity - K @ H) @ self.P_
        self.x_post_ = self.x_.copy()
        self.P_post_ = self.P_.copy()
        return self.state, self.P_.copy()

    def step(self, z, u=None, F=None, H=None, Q=None, R=None, B=None):
        self.predict(u=u, F=F, Q=Q, B=B)
        return self.update(z, H=H, R=R)

    def reset(self, x0=None, P0=None):
        if x0 is not None:
            self.x_ = np.asarray(x0, dtype=float).reshape(-1, 1)
        if P0 is not None:
            self.P_ = _as_2d_matrix(P0, "P0")
        self.x_prior_ = self.x_.copy()
        self.P_prior_ = self.P_.copy()
        self.x_post_ = self.x_.copy()
        self.P_post_ = self.P_.copy()
        return self.state
