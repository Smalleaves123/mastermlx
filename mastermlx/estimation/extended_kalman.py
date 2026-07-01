from __future__ import annotations

import numpy as np

try:
    from ._kalman_ops import kalman_predict as _cy_kalman_predict
    from ._kalman_ops import kalman_update_innovation as _cy_kalman_update_innovation
except ImportError:  # pragma: no cover - fallback when Cython extensions are unavailable
    _cy_kalman_predict = None
    _cy_kalman_update_innovation = None


def _as_2d_matrix(M, name):
    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    return M


class ExtendedKalmanFilter:
    """Extended Kalman filter for nonlinear systems."""

    def __init__(self, x0, P0, f, h, F_jac, H_jac, Q, R):
        self.x_ = np.asarray(x0, dtype=float).reshape(-1, 1)
        self.P_ = _as_2d_matrix(P0, "P0")
        self.f_ = f
        self.h_ = h
        self.F_jac_ = F_jac
        self.H_jac_ = H_jac
        self.Q_ = _as_2d_matrix(Q, "Q")
        self.R_ = _as_2d_matrix(R, "R")
        self.x_prior_ = self.x_.copy()
        self.P_prior_ = self.P_.copy()
        self.x_post_ = self.x_.copy()
        self.P_post_ = self.P_.copy()

    @property
    def state(self):
        return self.x_.ravel()

    def predict(self, u=None, dt=None):
        x_prev = self.x_.ravel()
        if dt is None:
            x_pred = self.f_(x_prev, u)
            F = self.F_jac_(x_prev, u)
        else:
            x_pred = self.f_(x_prev, u, dt)
            F = self.F_jac_(x_prev, u, dt)

        x_pred = np.asarray(x_pred, dtype=float).reshape(-1, 1)
        F = np.asarray(F, dtype=float)
        if _cy_kalman_predict is not None:
            self.x_, self.P_ = _cy_kalman_predict(x_pred, self.P_, F, self.Q_)
            self.x_ = np.asarray(self.x_, dtype=float).reshape(-1, 1)
            self.P_ = np.asarray(self.P_, dtype=float)
            self.x_prior_ = self.x_.copy()
            self.P_prior_ = self.P_.copy()
            return self.state, self.P_.copy()
        P_pred = F @ self.P_ @ F.T + self.Q_
        self.x_ = x_pred
        self.P_ = P_pred
        self.x_prior_ = x_pred.copy()
        self.P_prior_ = P_pred.copy()
        return self.state, self.P_.copy()

    def update(self, z, u=None, dt=None):
        x = self.x_.ravel()
        if dt is None:
            z_pred = self.h_(x, u)
            H = self.H_jac_(x, u)
        else:
            z_pred = self.h_(x, u, dt)
            H = self.H_jac_(x, u, dt)

        z = np.asarray(z, dtype=float).reshape(-1, 1)
        z_pred = np.asarray(z_pred, dtype=float).reshape(-1, 1)
        H = np.asarray(H, dtype=float)

        if _cy_kalman_update_innovation is not None:
            innovation = z - z_pred
            self.x_, self.P_ = _cy_kalman_update_innovation(self.x_, self.P_, innovation, H, self.R_)
            self.x_ = np.asarray(self.x_, dtype=float).reshape(-1, 1)
            self.P_ = np.asarray(self.P_, dtype=float)
            self.x_post_ = self.x_.copy()
            self.P_post_ = self.P_.copy()
            return self.state, self.P_.copy()

        y = z - z_pred
        S = H @ self.P_ @ H.T + self.R_
        K = self.P_ @ H.T @ np.linalg.solve(S, np.eye(S.shape[0], dtype=float))
        self.x_ = self.x_ + K @ y
        I = np.eye(self.P_.shape[0], dtype=float)
        self.P_ = (I - K @ H) @ self.P_
        self.x_post_ = self.x_.copy()
        self.P_post_ = self.P_.copy()
        return self.state, self.P_.copy()

    def step(self, z, u=None, dt=None):
        self.predict(u=u, dt=dt)
        return self.update(z, u=u, dt=dt)
