from __future__ import annotations

import numpy as np

from ..config import get_backend
from ._validation import as_matrix, as_square_matrix, as_vector, joseph_covariance

try:
    from ._kalman_ops import kalman_predict_covariance as _cy_kalman_predict_covariance
    from ._kalman_ops import kalman_update_innovation as _cy_kalman_update_innovation
except ImportError:  # pragma: no cover - fallback when Cython extensions are unavailable
    _cy_kalman_predict_covariance = None
    _cy_kalman_update_innovation = None


class ExtendedKalmanFilter:
    """Extended Kalman filter for nonlinear systems."""

    def __init__(self, x0, P0, f, h, F_jac, H_jac, Q, R):
        state = as_vector(x0, "x0")
        state_size = state.size
        self.x_ = state.reshape(-1, 1)
        self.P_ = as_square_matrix(P0, "P0", size=state_size)
        self.f_ = f
        self.h_ = h
        self.F_jac_ = F_jac
        self.H_jac_ = H_jac
        self.Q_ = as_square_matrix(Q, "Q", size=state_size)
        self.R_ = as_square_matrix(R, "R")
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

        state_size = self.x_.shape[0]
        x_pred = as_vector(x_pred, "predicted state", size=state_size).reshape(-1, 1)
        F = as_square_matrix(F, "F_jac", size=state_size)
        if get_backend() != "numpy" and _cy_kalman_predict_covariance is not None:
            self.x_ = x_pred
            self.P_ = np.asarray(
                _cy_kalman_predict_covariance(self.P_, F, self.Q_),
                dtype=float,
            )
            self.x_prior_ = self.x_.copy()
            self.P_prior_ = self.P_.copy()
            return self.state, self.P_.copy()
        P_pred = F @ self.P_ @ F.T + self.Q_
        P_pred = 0.5 * (P_pred + P_pred.T)
        self.x_ = x_pred
        self.P_ = P_pred
        self.x_prior_ = x_pred.copy()
        self.P_prior_ = P_pred.copy()
        return self.state, self.P_.copy()

    def update(self, z, u=None, dt=None, h=None, H_jac=None, R=None):
        x = self.x_.ravel()
        h_func = self.h_ if h is None else h
        H_jac_func = self.H_jac_ if H_jac is None else H_jac
        R = self.R_ if R is None else as_square_matrix(R, "R")
        if dt is None:
            z_pred = h_func(x, u)
            H = H_jac_func(x, u)
        else:
            z_pred = h_func(x, u, dt)
            H = H_jac_func(x, u, dt)

        state_size = self.x_.shape[0]
        z = as_vector(z, "z")
        z_pred = as_vector(z_pred, "predicted measurement", size=z.size)
        H = as_matrix(H, "H_jac", shape=(z.size, state_size))
        R = as_square_matrix(R, "R", size=z.size)
        z = z.reshape(-1, 1)
        z_pred = z_pred.reshape(-1, 1)

        if get_backend() != "numpy" and _cy_kalman_update_innovation is not None:
            innovation = z - z_pred
            self.x_, self.P_ = _cy_kalman_update_innovation(self.x_, self.P_, innovation, H, R)
            self.x_ = np.asarray(self.x_, dtype=float).reshape(-1, 1)
            self.P_ = np.asarray(self.P_, dtype=float)
            self.x_post_ = self.x_.copy()
            self.P_post_ = self.P_.copy()
            return self.state, self.P_.copy()

        y = z - z_pred
        S = H @ self.P_ @ H.T + R
        PHt = self.P_ @ H.T
        K = np.linalg.solve(S, PHt.T).T
        self.x_ = self.x_ + K @ y
        self.P_ = joseph_covariance(self.P_, K, H, R)
        self.x_post_ = self.x_.copy()
        self.P_post_ = self.P_.copy()
        return self.state, self.P_.copy()

    def step(self, z, u=None, dt=None, h=None, H_jac=None, R=None):
        self.predict(u=u, dt=dt)
        return self.update(z, u=u, dt=dt, h=h, H_jac=H_jac, R=R)

    def reset(self, x0=None, P0=None):
        state_size = self.x_.shape[0]
        if x0 is not None:
            self.x_ = as_vector(x0, "x0", size=state_size).reshape(-1, 1)
        if P0 is not None:
            self.P_ = as_square_matrix(P0, "P0", size=state_size)
        self.x_prior_ = self.x_.copy()
        self.P_prior_ = self.P_.copy()
        self.x_post_ = self.x_.copy()
        self.P_post_ = self.P_.copy()
        return self
