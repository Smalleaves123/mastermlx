from __future__ import annotations

import numpy as np

from ..estimation import ExtendedKalmanFilter


def wrap_angle(angle):
    angle = np.asarray(angle, dtype=float)
    return np.arctan2(np.sin(angle), np.cos(angle))


class PlanarPoseEKF:
    """Planar robot pose estimator with odometry and partial sensor updates.

    State: [x, y, yaw]
    Control input: [v, w]
    """

    def __init__(self, x0, P0, Q, R_heading, R_position=None, R_pose=None):
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        if x0.size != 3:
            raise ValueError("x0 must have shape (3,)")

        def motion_model(x, u, dt):
            x = np.asarray(x, dtype=float).reshape(-1)
            u = np.zeros(2, dtype=float) if u is None else np.asarray(u, dtype=float).reshape(-1)
            if u.size != 2:
                raise ValueError("odometry must have shape (2,) for [v, w]")
            v, w = u
            yaw = float(x[2])
            return np.array(
                [
                    x[0] + v * np.cos(yaw) * dt,
                    x[1] + v * np.sin(yaw) * dt,
                    wrap_angle(yaw + w * dt),
                ],
                dtype=float,
            )

        def motion_jacobian(x, u, dt):
            x = np.asarray(x, dtype=float).reshape(-1)
            u = np.zeros(2, dtype=float) if u is None else np.asarray(u, dtype=float).reshape(-1)
            if u.size != 2:
                raise ValueError("odometry must have shape (2,) for [v, w]")
            v, _ = u
            yaw = float(x[2])
            return np.array(
                [
                    [1.0, 0.0, -v * np.sin(yaw) * dt],
                    [0.0, 1.0, v * np.cos(yaw) * dt],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )

        self.filter_ = ExtendedKalmanFilter(
            x0=x0,
            P0=P0,
            f=motion_model,
            h=lambda x, u=None: np.asarray(x, dtype=float).reshape(-1, 1),
            F_jac=motion_jacobian,
            H_jac=lambda x, u=None: np.eye(3, dtype=float),
            Q=Q,
            R=np.eye(3, dtype=float),
        )
        self.R_heading_ = np.asarray(R_heading, dtype=float).reshape(1, 1)
        self.R_position_ = None if R_position is None else np.asarray(R_position, dtype=float).reshape(2, 2)
        self.R_pose_ = None if R_pose is None else np.asarray(R_pose, dtype=float).reshape(3, 3)

    @property
    def state(self):
        return self.filter_.state

    @property
    def covariance(self):
        return self.filter_.P_.copy()

    @property
    def pose(self):
        return self.filter_.state.copy()

    @property
    def position(self):
        return self.filter_.state[:2].copy()

    @property
    def yaw(self):
        return float(self.filter_.state[2])

    def predict(self, odometry, dt):
        return self.filter_.predict(u=odometry, dt=dt)

    def update_heading(self, yaw, R=None):
        R = self.R_heading_ if R is None else np.asarray(R, dtype=float).reshape(1, 1)

        def h(x, u=None, dt=None):
            return np.array([x[2]], dtype=float)

        def H_jac(x, u=None, dt=None):
            return np.array([[0.0, 0.0, 1.0]], dtype=float)

        return self.filter_.update(np.array([yaw], dtype=float), h=h, H_jac=H_jac, R=R)

    def update_position(self, position, R=None):
        position = np.asarray(position, dtype=float).reshape(-1)
        if position.size != 2:
            raise ValueError("position must have shape (2,)")
        if R is None:
            if self.R_position_ is None:
                raise ValueError("R_position is required for position updates")
            R = self.R_position_
        R = np.asarray(R, dtype=float).reshape(2, 2)

        def h(x, u=None, dt=None):
            return np.array([x[0], x[1]], dtype=float)

        def H_jac(x, u=None, dt=None):
            return np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=float,
            )

        return self.filter_.update(position, h=h, H_jac=H_jac, R=R)

    def update_pose(self, pose, R=None):
        pose = np.asarray(pose, dtype=float).reshape(-1)
        if pose.size == 2:
            return self.update_position(pose, R=R)
        if pose.size != 3:
            raise ValueError("pose must have shape (2,) or (3,)")
        if R is None:
            if self.R_pose_ is None:
                R = np.eye(3, dtype=float)
            else:
                R = self.R_pose_
        R = np.asarray(R, dtype=float).reshape(3, 3)

        def h(x, u=None, dt=None):
            return np.asarray(x, dtype=float).reshape(3)

        def H_jac(x, u=None, dt=None):
            return np.eye(3, dtype=float)

        return self.filter_.update(pose, h=h, H_jac=H_jac, R=R)

    def step(self, odometry, dt, heading=None, position=None, pose=None):
        result = self.predict(odometry, dt)
        if heading is not None:
            result = self.update_heading(heading)
        if position is not None:
            result = self.update_position(position)
        if pose is not None:
            result = self.update_pose(pose)
        return result

    def reset(self, x0=None, P0=None):
        return self.filter_.reset(x0=x0, P0=P0)
