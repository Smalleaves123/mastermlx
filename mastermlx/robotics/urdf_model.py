"""General spatial robot model backed by a serial URDF chain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constraints import clip_joint_values, validate_joint_limits
from .collision import (
    path_collision_free,
    path_collision_report,
    path_collision_summary,
    robot_collision_report,
)
from .results import RobotResult
from .urdf_parser import URDFSerialChain


def _pose_orientation_error(current, target):
    """Return a world-frame small-angle orientation error."""

    current_rotation = np.asarray(current, dtype=float)[:3, :3]
    target_rotation = np.asarray(target, dtype=float)[:3, :3]
    return 0.5 * (
        np.cross(current_rotation[:, 0], target_rotation[:, 0])
        + np.cross(current_rotation[:, 1], target_rotation[:, 1])
        + np.cross(current_rotation[:, 2], target_rotation[:, 2])
    )


@dataclass
class URDFRobotModel:
    """Kinematics model for a general serial spatial URDF chain.

    This model preserves URDF joint origins and axes, including non-zero RPY
    origins and joints that rotate or translate about arbitrary directions. It
    deliberately focuses on kinematics and task-space IK; the existing
    :class:`RobotModel` remains the compatibility path for DH dynamics and
    workcell features.
    """

    chain: URDFSerialChain
    name: str = "robot"
    base: np.ndarray | None = None
    tool: np.ndarray | None = None
    joint_limits: np.ndarray | None = None

    def __post_init__(self):
        if not isinstance(self.chain, URDFSerialChain):
            raise TypeError("chain must be a URDFSerialChain")
        if self.base is not None:
            self.base = np.asarray(self.base, dtype=float)
            if self.base.shape != (4, 4):
                raise ValueError("base must have shape (4, 4)")
        if self.tool is not None:
            self.tool = np.asarray(self.tool, dtype=float)
            if self.tool.shape != (4, 4):
                raise ValueError("tool must have shape (4, 4)")
        limits = self.chain.joint_limits if self.joint_limits is None else self.joint_limits
        self.joint_limits = validate_joint_limits(limits, self.n_joints)

    @classmethod
    def from_urdf(
        cls,
        xml_text,
        *,
        name=None,
        base_link=None,
        tip_link=None,
        base=None,
        tool=None,
        joint_limits=None,
    ):
        chain = URDFSerialChain.from_urdf(
            xml_text, base_link=base_link, tip_link=tip_link
        )
        return cls(
            chain=chain,
            name=chain.tip_link if name is None else str(name),
            base=base,
            tool=tool,
            joint_limits=joint_limits,
        )

    @property
    def n_joints(self):
        return self.chain.n_joints

    @property
    def joint_names(self):
        return self.chain.joint_names

    @property
    def joint_types(self):
        return self.chain.joint_types

    def default_joint_values(self):
        if self.joint_limits is None:
            return np.zeros(self.n_joints, dtype=float)
        zero = np.zeros(self.n_joints, dtype=float)
        if np.all((zero >= self.joint_limits[:, 0]) & (zero <= self.joint_limits[:, 1])):
            return zero
        return np.mean(self.joint_limits, axis=1)

    def validate_joint_values(self, joint_values=None, *, check_limits=False):
        values = self.chain.validate_joint_values(joint_values)
        if check_limits:
            if self.joint_limits is not None and (
                np.any(values < self.joint_limits[:, 0])
                or np.any(values > self.joint_limits[:, 1])
            ):
                raise ValueError("joint_values exceeds configured joint_limits")
        return values

    def clip_joint_values(self, joint_values):
        values = self.validate_joint_values(joint_values)
        return clip_joint_values(values, self.joint_limits)

    def fk(self, joint_values=None, *, return_all=False):
        values = self.default_joint_values() if joint_values is None else joint_values
        values = self.validate_joint_values(values, check_limits=self.joint_limits is not None)
        return self.chain.forward_kinematics(
            values, base=self.base, tool=self.tool, return_all=return_all
        )

    def forward_kinematics(self, joint_values=None, *, return_all=False):
        return self.fk(joint_values=joint_values, return_all=return_all)

    def fk_batch(self, joint_values):
        values = np.asarray(joint_values, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_joints:
            raise ValueError(f"joint_values must have shape (n_samples, {self.n_joints})")
        if self.joint_limits is not None:
            for row in values:
                self.validate_joint_values(row, check_limits=True)
        return self.chain.forward_kinematics_batch(values, base=self.base, tool=self.tool)

    def positions(self, joint_values=None):
        values = self.default_joint_values() if joint_values is None else joint_values
        self.validate_joint_values(values, check_limits=self.joint_limits is not None)
        return self.chain.positions(values, base=self.base, tool=self.tool)

    def frame_positions(self, joint_values=None):
        """Return the base, intermediate, and tool frame positions."""

        return self.positions(joint_values=joint_values)

    def frame_positions_batch(self, joint_values):
        """Return all chain frame positions for a batch of configurations."""

        values = np.asarray(joint_values, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_joints:
            raise ValueError(f"joint_values must have shape (n_samples, {self.n_joints})")
        if self.joint_limits is not None:
            for row in values:
                self.validate_joint_values(row, check_limits=True)
        return np.asarray(
            [self.frame_positions(row) for row in values], dtype=float
        )

    def collision_report(
        self,
        joint_values=None,
        obstacles=(),
        *,
        link_radius=0.0,
        occupancy_grid=None,
    ):
        """Return geometric and optional occupancy-grid collision diagnostics."""

        return robot_collision_report(
            self,
            joint_values,
            obstacles,
            link_radius=link_radius,
            occupancy_grid=occupancy_grid,
        )

    def path_collision_free(
        self,
        joint_path,
        obstacles=(),
        *,
        clearance=0.0,
        link_radius=0.0,
        interpolation_step=0.05,
        occupancy_grid=None,
    ):
        """Return whether a joint path clears geometry and an optional voxel map."""

        return path_collision_free(
            self,
            joint_path,
            obstacles,
            clearance=clearance,
            link_radius=link_radius,
            interpolation_step=interpolation_step,
            occupancy_grid=occupancy_grid,
        )

    def path_collision_summary(
        self,
        joint_path,
        obstacles=(),
        *,
        link_radius=0.0,
        interpolation_step=0.05,
        occupancy_grid=None,
    ):
        """Return batched collision and clearance data for a joint path."""

        return path_collision_summary(
            self,
            joint_path,
            obstacles,
            link_radius=link_radius,
            interpolation_step=interpolation_step,
            occupancy_grid=occupancy_grid,
        )

    def path_collision_report(
        self,
        joint_path,
        obstacles=(),
        *,
        link_radius=0.0,
        interpolation_step=0.05,
        occupancy_grid=None,
    ):
        """Return detailed collision reports for a joint path."""

        return path_collision_report(
            self,
            joint_path,
            obstacles,
            link_radius=link_radius,
            interpolation_step=interpolation_step,
            occupancy_grid=occupancy_grid,
        )

    def jacobian(self, joint_values=None):
        values = self.default_joint_values() if joint_values is None else joint_values
        values = self.validate_joint_values(values, check_limits=self.joint_limits is not None)
        return self.chain.geometric_jacobian(values, base=self.base, tool=self.tool)

    def geometric_jacobian(self, joint_values=None):
        return self.jacobian(joint_values=joint_values)

    def jacobian_batch(self, joint_values):
        values = np.asarray(joint_values, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_joints:
            raise ValueError(f"joint_values must have shape (n_samples, {self.n_joints})")
        if self.joint_limits is not None:
            for row in values:
                self.validate_joint_values(row, check_limits=True)
        return self.chain.geometric_jacobian_batch(values, base=self.base, tool=self.tool)

    def end_effector_velocity(self, joint_values, joint_velocities, *, translational=False):
        q = self.validate_joint_values(joint_values, check_limits=self.joint_limits is not None)
        qd = self.validate_joint_values(joint_velocities)
        jacobian = self.jacobian(q)
        return (jacobian[:3] if translational else jacobian) @ qd

    def differential_ik(
        self,
        joint_values,
        task_velocity,
        *,
        damping=1e-4,
        translational=False,
    ):
        q = self.validate_joint_values(joint_values, check_limits=self.joint_limits is not None)
        velocity = np.asarray(task_velocity, dtype=float).reshape(-1)
        expected = 3 if translational else 6
        if velocity.size != expected:
            raise ValueError(f"task_velocity must contain {expected} values")
        damping = float(damping)
        if not np.isfinite(damping) or damping < 0.0:
            raise ValueError("damping must be finite and non-negative")
        jacobian = self.jacobian(q)
        if translational:
            jacobian = jacobian[:3]
        gram = jacobian @ jacobian.T
        regularized = gram + damping**2 * np.eye(gram.shape[0], dtype=float)
        return jacobian.T @ np.linalg.solve(regularized, velocity)

    def inverse_kinematics(
        self,
        target,
        joint_values=None,
        *,
        max_iter=200,
        tol=1e-6,
        damping=1e-4,
        step_size=1.0,
        position_weight=1.0,
        orientation_weight=1.0,
        return_info=False,
    ):
        """Solve a position or full-pose target with damped least squares."""

        target = np.asarray(target, dtype=float)
        position_only = target.shape == (3,)
        if not position_only and target.shape != (4, 4):
            raise ValueError("target must have shape (3,) or (4, 4)")
        if not np.all(np.isfinite(target)):
            raise ValueError("target must contain only finite values")
        max_iter = int(max_iter)
        tol = float(tol)
        damping = float(damping)
        step_size = float(step_size)
        if max_iter < 1 or tol <= 0.0 or damping < 0.0 or step_size <= 0.0:
            raise ValueError("max_iter, tol, damping, and step_size must be valid positive values")
        if position_weight <= 0.0 or orientation_weight <= 0.0:
            raise ValueError("position_weight and orientation_weight must be positive")

        q = self.default_joint_values() if joint_values is None else joint_values
        q = self.validate_joint_values(q, check_limits=self.joint_limits is not None).copy()
        target_position = target if position_only else target[:3, 3]
        target_pose = None if position_only else target
        converged = False
        error_norm = np.inf
        iterations = 0
        for iterations in range(1, max_iter + 1):
            current = self.fk(q)
            error = target_position - current[:3, 3]
            jacobian = self.jacobian(q)[:3]
            if target_pose is not None:
                orientation_error = _pose_orientation_error(current, target_pose)
                error = np.concatenate(
                    [position_weight * error, orientation_weight * orientation_error]
                )
                full_jacobian = self.jacobian(q)
                full_jacobian = np.vstack(
                    [position_weight * full_jacobian[:3], orientation_weight * full_jacobian[3:]]
                )
            else:
                full_jacobian = position_weight * jacobian
            error_norm = float(np.linalg.norm(error))
            if error_norm <= tol:
                converged = True
                break
            gram = full_jacobian @ full_jacobian.T
            regularized = gram + damping**2 * np.eye(gram.shape[0], dtype=float)
            q += step_size * full_jacobian.T @ np.linalg.solve(regularized, error)
            if self.joint_limits is not None:
                q = self.clip_joint_values(q)

        if return_info:
            return RobotResult({
                "joint_values": q,
                "converged": converged,
                "iterations": iterations,
                "error_norm": error_norm,
                "position_only": position_only,
            })
        if not converged:
            raise RuntimeError(
                f"inverse kinematics did not converge within {max_iter} iterations "
                f"(error_norm={error_norm:.3e})"
            )
        return q


__all__ = ["URDFRobotModel"]
