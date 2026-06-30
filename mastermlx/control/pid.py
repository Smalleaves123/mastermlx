from __future__ import annotations

import numpy as np


class PIDController:
    """Discrete PID controller with optional output limits and anti-windup."""

    def __init__(self, kp=1.0, ki=0.0, kd=0.0, setpoint=0.0, output_limits=None):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.setpoint = np.asarray(setpoint, dtype=float)
        self.output_limits = output_limits
        self.integral_ = None
        self.prev_error_ = None
        self.reset()

    def reset(self):
        self.integral_ = np.zeros_like(self.setpoint, dtype=float)
        self.prev_error_ = None

    def _clip(self, u):
        if self.output_limits is None:
            return u
        lo, hi = self.output_limits
        if lo is not None:
            u = np.maximum(u, lo)
        if hi is not None:
            u = np.minimum(u, hi)
        return u

    def update(self, measurement, dt=1.0, setpoint=None):
        measurement = np.asarray(measurement, dtype=float)
        target = self.setpoint if setpoint is None else np.asarray(setpoint, dtype=float)
        error = target - measurement
        dt = float(dt)
        if dt <= 0:
            raise ValueError("dt must be positive")

        if self.integral_.shape != error.shape:
            self.integral_ = np.zeros_like(error, dtype=float)

        self.integral_ = self.integral_ + error * dt
        derivative = np.zeros_like(error)
        if self.prev_error_ is not None:
            derivative = (error - self.prev_error_) / dt
        u = self.kp * error + self.ki * self.integral_ + self.kd * derivative
        u = self._clip(u)

        if self.output_limits is not None and self.ki != 0.0:
            lo, hi = self.output_limits
            if lo is not None or hi is not None:
                unclipped = self.kp * error + self.ki * self.integral_ + self.kd * derivative
                if np.any(u != unclipped):
                    self.integral_ = self.integral_ - error * dt

        self.prev_error_ = error
        return u
