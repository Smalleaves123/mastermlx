from __future__ import annotations

import numpy as np

from ..config import get_backend
from ._validation import as_matrix, as_square_matrix, as_vector, joseph_covariance

try:
    from ._kalman_ops import kalman_predict as _cy_kalman_predict
    from ._kalman_ops import kalman_update as _cy_kalman_update
except ImportError:  # pragma: no cover - fallback when Cython extensions are unavailable
    _cy_kalman_predict = None
    _cy_kalman_update = None


class KalmanFilter:
    """Linear Kalman filter for discrete-time systems.

    State model:
        x_k = F x_{k-1} + B u_k + w_k
        z_k = H x_k + v_k
    """

    def __init__(self, x0, P0, F, H, Q, R, B=None):
        state = as_vector(x0, "x0")
        state_size = state.size
        self.x_ = state.reshape(-1, 1)
        self.P_ = as_square_matrix(P0, "P0", size=state_size)
        self.F_ = as_square_matrix(F, "F", size=state_size)
        self.Q_ = as_square_matrix(Q, "Q", size=state_size)
        self.H_ = as_matrix(H, "H")
        if self.H_.shape[0] < 1 or self.H_.shape[1] != state_size:
            raise ValueError(f"H must have shape (n_measurements, {state_size})")
        self.R_ = as_square_matrix(R, "R", size=self.H_.shape[0])
        self.B_ = None if B is None else as_matrix(B, "B")
        if self.B_ is not None and (
            self.B_.shape[0] != state_size or self.B_.shape[1] < 1
        ):
            raise ValueError(f"B must have shape ({state_size}, n_controls)")
        self.x_prior_ = self.x_.copy()
        self.P_prior_ = self.P_.copy()
        self.x_post_ = self.x_.copy()
        self.P_post_ = self.P_.copy()

    @property
    def state(self):
        return self.x_.ravel()

    def predict(self, u=None, F=None, Q=None, B=None):
        state_size = self.x_.shape[0]
        F = self.F_ if F is None else as_square_matrix(F, "F", size=state_size)
        Q = self.Q_ if Q is None else as_square_matrix(Q, "Q", size=state_size)
        B = self.B_ if B is None else as_matrix(B, "B")
        if B is not None and (B.shape[0] != state_size or B.shape[1] < 1):
            raise ValueError(f"B must have shape ({state_size}, n_controls)")
        if u is not None:
            if B is None:
                raise ValueError("Control input provided but no control matrix B is available")
            u = as_vector(u, "u", size=B.shape[1])

        if get_backend() != "numpy" and _cy_kalman_predict is not None:
            self.x_, self.P_ = _cy_kalman_predict(self.x_, self.P_, F, Q, B=B, u=u)
            self.x_ = np.asarray(self.x_, dtype=float).reshape(-1, 1)
            self.P_ = np.asarray(self.P_, dtype=float)
            self.x_prior_ = self.x_.copy()
            self.P_prior_ = self.P_.copy()
            return self.state, self.P_.copy()

        x = F @ self.x_
        if u is not None:
            x = x + B @ u.reshape(-1, 1)

        P = F @ self.P_ @ F.T + Q
        P = 0.5 * (P + P.T)
        self.x_prior_ = x
        self.P_prior_ = P
        self.x_ = x
        self.P_ = P
        return self.state, self.P_.copy()

    def update(self, z, H=None, R=None):
        state_size = self.x_.shape[0]
        H = self.H_ if H is None else as_matrix(H, "H")
        if H.shape[0] < 1 or H.shape[1] != state_size:
            raise ValueError(f"H must have shape (n_measurements, {state_size})")
        R = self.R_ if R is None else as_square_matrix(R, "R", size=H.shape[0])
        if R.shape != (H.shape[0], H.shape[0]):
            raise ValueError(f"R must have shape ({H.shape[0]}, {H.shape[0]})")
        z = as_vector(z, "z", size=H.shape[0])
        if get_backend() != "numpy" and _cy_kalman_update is not None:
            self.x_, self.P_ = _cy_kalman_update(self.x_, self.P_, z, H, R)
            self.x_ = np.asarray(self.x_, dtype=float).reshape(-1, 1)
            self.P_ = np.asarray(self.P_, dtype=float)
            self.x_post_ = self.x_.copy()
            self.P_post_ = self.P_.copy()
            return self.state, self.P_.copy()

        z = z.reshape(-1, 1)

        y = z - H @ self.x_
        S = H @ self.P_ @ H.T + R
        PHt = self.P_ @ H.T
        K = np.linalg.solve(S, PHt.T).T
        self.x_ = self.x_ + K @ y
        self.P_ = joseph_covariance(self.P_, K, H, R)
        self.x_post_ = self.x_.copy()
        self.P_post_ = self.P_.copy()
        return self.state, self.P_.copy()

    def step(self, z, u=None, F=None, H=None, Q=None, R=None, B=None):
        self.predict(u=u, F=F, Q=Q, B=B)
        return self.update(z, H=H, R=R)

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
        return self.state
