"""Hardware-neutral joint controller contracts and implementations."""

from __future__ import annotations

import numpy as np

from ..base import BaseResult
from .mpc import LinearMPC


class ControllerStatus(BaseResult):
    """Mapping-compatible snapshot of a controller execution state."""


def _vector(value, size, name):
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(size, float(array), dtype=float)
    else:
        array = array.reshape(-1)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite scalar or shape ({size},)")
    return array


def _output_limits(limits, size):
    if limits is None:
        return None
    if len(limits) != 2:
        raise ValueError("output_limits must be a (lower, upper) pair")
    lower = None if limits[0] is None else _vector(limits[0], size, "output_limits lower")
    upper = None if limits[1] is None else _vector(limits[1], size, "output_limits upper")
    if lower is not None and upper is not None and np.any(lower > upper):
        raise ValueError("output_limits lower values must not exceed upper values")
    return lower, upper


def _at_limits(output, limits):
    """Return whether a command is on either side of a configured box."""

    lower, upper = limits
    tolerance = 1e-12 * np.maximum(1.0, np.abs(output))
    if lower is not None and np.any(output <= lower + tolerance):
        return True
    if upper is not None and np.any(output >= upper - tolerance):
        return True
    return False


class Controller:
    """Minimal protocol for a stateful, hardware-neutral controller."""

    name = "controller"
    command_type = "acceleration"

    def __init__(self, n_joints):
        self.n_joints = int(n_joints)
        if self.n_joints < 1:
            raise ValueError("n_joints must be positive")
        self.reset()

    def reset(self, state=None):
        """Reset controller state before a new execution."""

        del state
        self.steps_ = 0
        self.last_output_ = None
        self.saturated_ = False
        self.fault_ = None

    def _state(self, state):
        value = np.asarray(state, dtype=float).reshape(-1)
        if value.shape != (2 * self.n_joints,) or not np.all(np.isfinite(value)):
            raise ValueError(f"state must be a finite vector with shape ({2 * self.n_joints},)")
        return value[: self.n_joints], value[self.n_joints :]

    def _reference(self, reference):
        return _vector(reference, self.n_joints, "reference")

    def _record(self, output, *, saturated=False):
        output = _vector(output, self.n_joints, "controller output")
        self.steps_ += 1
        self.last_output_ = output.copy()
        self.saturated_ = bool(saturated)
        return output

    def update(self, reference, state, dt):
        """Compute one control command for a joint position reference."""

        raise NotImplementedError

    def status(self):
        """Return a serializable execution snapshot."""

        return ControllerStatus({
            "name": self.name,
            "command_type": self.command_type,
            "n_joints": self.n_joints,
            "steps": self.steps_,
            "last_output": None if self.last_output_ is None else self.last_output_.copy(),
            "saturated": bool(self.saturated_),
            "fault": self.fault_,
        })


class JointPDController(Controller):
    """Joint-space PD controller with optional command limits."""

    name = "pd"

    def __init__(self, n_joints, kp=4.0, kd=0.4, *, output_limits=None):
        self.kp = _vector(kp, int(n_joints), "kp")
        self.kd = _vector(kd, int(n_joints), "kd")
        if np.any(self.kp < 0.0) or np.any(self.kd < 0.0):
            raise ValueError("kp and kd must be non-negative")
        self.output_limits = _output_limits(output_limits, int(n_joints))
        super().__init__(n_joints)

    def update(self, reference, state, dt):
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        q, qd = self._state(state)
        target = self._reference(reference)
        raw = self.kp * (target - q) - self.kd * qd
        output = raw.copy()
        if self.output_limits is not None:
            lower, upper = self.output_limits
            if lower is not None:
                output = np.maximum(output, lower)
            if upper is not None:
                output = np.minimum(output, upper)
        return self._record(output, saturated=not np.array_equal(raw, output))


class JointMPCController(Controller):
    """Linear MPC controller for the virtual second-order joint model."""

    name = "mpc"

    def __init__(self, n_joints, dt=0.1, *, damping=0.0, output_limits=None, mpc_kwargs=None):
        self.dt = float(dt)
        self.damping = float(damping)
        if self.dt <= 0.0 or not np.isfinite(self.dt):
            raise ValueError("dt must be positive and finite")
        if self.damping < 0.0 or not np.isfinite(self.damping):
            raise ValueError("damping must be non-negative and finite")
        self.output_limits = _output_limits(output_limits, int(n_joints))
        identity = np.eye(int(n_joints), dtype=float)
        damping_factor = 1.0 - self.dt * self.damping
        A = np.block([
            [identity, self.dt * damping_factor * identity],
            [np.zeros_like(identity), damping_factor * identity],
        ])
        B = np.vstack([self.dt**2 * identity, self.dt * identity])
        options = {} if mpc_kwargs is None else dict(mpc_kwargs)
        options.setdefault("horizon", 10)
        options.setdefault("Q", np.diag(np.concatenate([
            np.full(int(n_joints), 10.0), np.ones(int(n_joints))
        ])))
        options.setdefault("R", 0.1 * identity)
        options.setdefault("Qf", options["Q"])
        if self.output_limits is not None and "u_bounds" not in options:
            options["u_bounds"] = self.output_limits
        self._controller = LinearMPC(A, B, **options)
        super().__init__(n_joints)

    def update(self, reference, state, dt):
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        if not np.isclose(dt, self.dt):
            raise ValueError("dt must match the controller sampling period")
        q, qd = self._state(state)
        target = self._reference(reference)
        raw = np.asarray(
            self._controller.control(
                np.concatenate([q, qd]),
                x_ref=np.concatenate([target, np.zeros(self.n_joints)]),
            ),
            dtype=float,
        )
        output = raw.copy()
        if self.output_limits is not None:
            lower, upper = self.output_limits
            if lower is not None:
                output = np.maximum(output, lower)
            if upper is not None:
                output = np.minimum(output, upper)
        saturated = not np.array_equal(raw, output)
        if self._controller.u_bounds_ is not None:
            lower, upper = self._controller.u_bounds_
            saturated = saturated or _at_limits(output, (lower, upper))
        return self._record(output, saturated=saturated)

    def status(self):
        result = super().status()
        result["qp_converged"] = bool(self._controller.qp_converged_)
        result["qp_iterations"] = int(self._controller.last_qp_iterations_)
        return result


class ComputedTorqueController(Controller):
    """Dynamics-compensated joint controller for DH or URDF robots."""

    name = "computed_torque"
    command_type = "torque"

    def __init__(
        self,
        robot,
        *,
        kp=25.0,
        kd=8.0,
        gravity=(0.0, 0.0, -9.81),
        link_inertias=None,
        output_limits=None,
    ):
        self.robot = robot
        self.kp = _vector(kp, robot.n_joints, "kp")
        self.kd = _vector(kd, robot.n_joints, "kd")
        if np.any(self.kp <= 0.0) or np.any(self.kd <= 0.0):
            raise ValueError("kp and kd must be positive")
        self.gravity = _vector(gravity, 3, "gravity")
        self.link_inertias = link_inertias
        self.output_limits = _output_limits(output_limits, robot.n_joints)
        super().__init__(robot.n_joints)

    def update(self, reference, state, dt):
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        q, qd = self._state(state)
        target = self._reference(reference)
        qdd = self.kp * (target - q) - self.kd * qd
        if callable(getattr(self.robot, "computed_torque_control", None)):
            raw = self.robot.computed_torque_control(
                q,
                qd,
                target,
                desired_accelerations=qdd,
                kp=self.kp,
                kd=self.kd,
                gravity=self.gravity,
                link_inertias=self.link_inertias,
            )
        else:
            raw = self.robot.inverse_dynamics_batch(
                q[None, :],
                qd[None, :],
                qdd[None, :],
                gravity=self.gravity,
                include_coriolis=True,
            )[0]
        raw = np.asarray(raw, dtype=float).reshape(-1)
        output = raw.copy()
        saturated = False
        if self.output_limits is not None:
            lower, upper = self.output_limits
            if lower is not None:
                output = np.maximum(output, lower)
            if upper is not None:
                output = np.minimum(output, upper)
            saturated = not np.array_equal(raw, output)
        return self._record(output, saturated=saturated)


__all__ = [
    "Controller",
    "ControllerStatus",
    "ComputedTorqueController",
    "JointMPCController",
    "JointPDController",
]
