"""Stable result containers for the public robotics API."""

from __future__ import annotations

from typing import Any

import numpy as np


class RobotResult(dict[str, Any]):
    """Dictionary-compatible result with attribute access.

    Mapping access remains available for compatibility with earlier releases,
    while attribute access makes nested robotics workflows easier to read::

        result["joint_path"]
        result.joint_path
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def as_dict(self) -> dict[str, Any]:
        """Return a shallow plain-dictionary view of the result."""

        return dict(self)


class JointTrajectory(RobotResult):
    """Validated joint trajectory returned by workcell retiming.

    The object is still a ``dict`` subclass, so existing indexing and JSON
    export code continues to work.  Required arrays are exposed as named
    attributes and basic dimensions are checked at construction time.
    """

    def __init__(
        self,
        *,
        time,
        position,
        velocity,
        acceleration,
        jerk,
        durations,
        path,
        velocity_limits=None,
        acceleration_limits=None,
        jerk_limits=None,
    ):
        values = {
            "time": np.asarray(time, dtype=float),
            "position": np.asarray(position, dtype=float),
            "velocity": np.asarray(velocity, dtype=float),
            "acceleration": np.asarray(acceleration, dtype=float),
            "jerk": np.asarray(jerk, dtype=float),
            "durations": np.asarray(durations, dtype=float).reshape(-1),
            "path": np.asarray(path, dtype=float),
            "velocity_limits": None if velocity_limits is None else np.asarray(velocity_limits, dtype=float),
            "acceleration_limits": None
            if acceleration_limits is None
            else np.asarray(acceleration_limits, dtype=float),
            "jerk_limits": None if jerk_limits is None else np.asarray(jerk_limits, dtype=float),
        }
        self._validate(values)
        super().__init__(values)

    @staticmethod
    def _validate(values):
        position = values["position"]
        if position.ndim != 2 or position.shape[0] < 2:
            raise ValueError("position must have shape (n_samples, n_joints) with at least two samples")
        n_samples, n_joints = position.shape
        if values["time"].shape != (n_samples,):
            raise ValueError("time must have one value per position sample")
        for name in ("velocity", "acceleration", "jerk"):
            if values[name].shape != position.shape:
                raise ValueError(f"{name} must have the same shape as position")
        if values["path"].ndim != 2 or values["path"].shape[1] != n_joints:
            raise ValueError("path must have shape (n_waypoints, n_joints)")
        if values["durations"].size != values["path"].shape[0] - 1:
            raise ValueError("durations must have one value per path segment")
        for name in ("velocity_limits", "acceleration_limits", "jerk_limits"):
            limits = values[name]
            if limits is not None and limits.shape != (n_joints,):
                raise ValueError(f"{name} must have one value per joint")
        if np.any(np.diff(values["time"]) <= 0.0):
            raise ValueError("time must be strictly increasing")
        if np.any(values["durations"] <= 0.0):
            raise ValueError("durations must be positive")
        finite_arrays = [values[name] for name in ("time", "position", "velocity", "acceleration", "jerk", "durations", "path")]
        finite_arrays.extend(values[name] for name in ("velocity_limits", "acceleration_limits", "jerk_limits") if values[name] is not None)
        if not all(np.all(np.isfinite(array)) for array in finite_arrays):
            raise ValueError("trajectory arrays must contain only finite values")

    @property
    def n_samples(self) -> int:
        return int(self["position"].shape[0])

    @property
    def n_joints(self) -> int:
        return int(self["position"].shape[1])

    @property
    def duration(self) -> float:
        return float(self["time"][-1] - self["time"][0])


__all__ = ["JointTrajectory", "RobotResult"]
