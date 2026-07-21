from __future__ import annotations

import numpy as np


class PIDController:
    """Discrete PID controller with output and integral anti-windup limits.

    ``derivative_on_measurement`` avoids a derivative kick when the setpoint
    changes abruptly by differentiating the measurement instead of the error.
    """

    def __init__(
        self,
        kp=1.0,
        ki=0.0,
        kd=0.0,
        setpoint=0.0,
        output_limits=None,
        integral_limits=None,
        *,
        derivative_on_measurement=False,
    ):
        for name, value in (("kp", kp), ("ki", ki), ("kd", kd)):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.setpoint = np.asarray(setpoint, dtype=float)
        if not np.all(np.isfinite(self.setpoint)):
            raise ValueError("setpoint must contain only finite values")
        self.output_limits = self._normalize_limits(output_limits, "output_limits")
        self.integral_limits = self._normalize_limits(integral_limits, "integral_limits")
        self.derivative_on_measurement = bool(derivative_on_measurement)
        self.integral_ = None
        self.prev_error_ = None
        self.prev_measurement_ = None
        self.last_output_ = None
        self.reset()

    @staticmethod
    def _normalize_limits(limits, name):
        if limits is None:
            return None
        if len(limits) != 2:
            raise ValueError(f"{name} must be a (lower, upper) pair")
        normalized: list[np.ndarray | None] = []
        for value in limits:
            if value is None:
                normalized.append(None)
                continue
            value = np.asarray(value, dtype=float)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values")
            normalized.append(value.copy())
        lower, upper = normalized
        if lower is not None and upper is not None and np.any(lower > upper):
            raise ValueError(f"{name} lower bound must not exceed upper bound")
        return lower, upper

    def reset(self):
        self.integral_ = np.zeros_like(self.setpoint, dtype=float)
        self.prev_error_ = None
        self.prev_measurement_ = None
        self.last_output_ = None

    def _clip(self, u):
        if self.output_limits is None:
            return u
        lo, hi = self.output_limits
        if lo is not None:
            u = np.maximum(u, lo)
        if hi is not None:
            u = np.minimum(u, hi)
        return u

    @staticmethod
    def _apply_limits(value, limits):
        if limits is None:
            return value
        lo, hi = limits
        if lo is not None:
            value = np.maximum(value, lo)
        if hi is not None:
            value = np.minimum(value, hi)
        return value

    def update(self, measurement, dt=1.0, setpoint=None):
        measurement = np.asarray(measurement, dtype=float)
        target = self.setpoint if setpoint is None else np.asarray(setpoint, dtype=float)
        if not np.all(np.isfinite(measurement)) or not np.all(np.isfinite(target)):
            raise ValueError("measurement and setpoint must contain only finite values")
        if measurement.shape != target.shape and measurement.ndim != 0 and target.ndim != 0:
            raise ValueError("measurement and setpoint must have compatible shapes")
        error = target - measurement
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")

        integral = self.integral_
        if integral is None or integral.shape != error.shape:
            self.integral_ = np.zeros_like(error, dtype=float)

        self.integral_ = self.integral_ + error * dt
        self.integral_ = self._apply_limits(self.integral_, self.integral_limits)
        derivative = np.zeros_like(error)
        if self.derivative_on_measurement and self.prev_measurement_ is not None:
            derivative = -(measurement - self.prev_measurement_) / dt
        elif not self.derivative_on_measurement and self.prev_error_ is not None:
            derivative = (error - self.prev_error_) / dt
        u = self.kp * error + self.ki * self.integral_ + self.kd * derivative
        u = self._clip(u)

        if self.output_limits is not None and self.ki != 0.0:
            lo, hi = self.output_limits
            if lo is not None or hi is not None:
                unclipped = self.kp * error + self.ki * self.integral_ + self.kd * derivative
                if np.any(u != unclipped):
                    self.integral_ = self.integral_ - error * dt
                    self.integral_ = self._apply_limits(self.integral_, self.integral_limits)

        self.prev_error_ = error
        self.prev_measurement_ = measurement.copy()
        self.last_output_ = np.asarray(u, dtype=float).copy()
        return u
