from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics import (
    DHLink,
    chain_positions,
    forward_kinematics,
    forward_kinematics_batch,
    inverse_kinematics,
    inverse_kinematics_batch,
)
from .constraints import check_joint_limits, clip_joint_values, joint_limit_violation, validate_joint_limits
from .jacobian import geometric_jacobian, geometric_jacobian_batch
from .results import RobotResult
from .urdf_parser import parse_urdf, urdf_to_dh_chain
from .visualizer import plot_chain


@dataclass
class RobotModel:
    """Lightweight serial robot model wrapper."""

    links: list[DHLink]
    name: str = "robot"
    base: np.ndarray | None = None
    tool: np.ndarray | None = None
    joint_limits: np.ndarray | None = None

    def __post_init__(self):
        self.links = [link if isinstance(link, DHLink) else DHLink(**link) for link in self.links]
        self.joint_limits = validate_joint_limits(self.joint_limits, len(self.links))

    @property
    def n_joints(self):
        """Number of actuated joints in the serial chain."""

        return len(self.links)

    def default_joint_values(self):
        """Return a safe default configuration for the model."""

        if self.joint_limits is None:
            return np.zeros(self.n_joints, dtype=float)
        zero = np.zeros(self.n_joints, dtype=float)
        if np.all((zero >= self.joint_limits[:, 0]) & (zero <= self.joint_limits[:, 1])):
            return zero
        return np.mean(self.joint_limits, axis=1)

    def validate_joint_values(self, joint_values, *, batch=False, check_limits=False):
        """Validate and normalize one or more joint configurations."""

        values = np.asarray(joint_values, dtype=float)
        if batch:
            if values.ndim != 2 or values.shape[1] != self.n_joints:
                raise ValueError(f"joint_values must have shape (n_samples, {self.n_joints})")
        elif values.reshape(-1).size != self.n_joints:
            raise ValueError(f"joint_values must contain {self.n_joints} values")
        if not np.all(np.isfinite(values)):
            raise ValueError("joint_values must contain only finite values")
        if check_limits:
            check_joint_limits(values, self.joint_limits)
        return values if batch else values.reshape(self.n_joints)

    def clip_joint_values(self, joint_values):
        """Project one or more joint configurations into model limits."""

        values = self.validate_joint_values(joint_values, batch=np.asarray(joint_values).ndim == 2)
        return clip_joint_values(values, self.joint_limits)

    def joint_limit_violation(self, joint_values):
        """Return maximum position-limit violation per configuration."""

        values = np.asarray(joint_values, dtype=float)
        if values.ndim == 1:
            values = self.validate_joint_values(values)
        else:
            values = self.validate_joint_values(values, batch=True)
        return joint_limit_violation(values, self.joint_limits)

    @classmethod
    def from_urdf(cls, xml_text, *, name=None, base_link=None, tip_link=None):
        links, joint_limits = urdf_to_dh_chain(
            xml_text,
            base_link=base_link,
            tip_link=tip_link,
            return_limits=True,
        )
        if name is None:
            parsed_links, _ = parse_urdf(xml_text)
            name = "robot" if not parsed_links else parsed_links[0].name
        return cls(links=links, name=name, joint_limits=joint_limits)

    @classmethod
    def from_dh(cls, links, *, name="robot", base=None, tool=None, joint_limits=None):
        """Build a robot model from DH links or link-like dictionaries."""

        return cls(
            links=[link if isinstance(link, DHLink) else DHLink(**link) for link in links],
            name=name,
            base=base,
            tool=tool,
            joint_limits=joint_limits,
        )

    def fk(self, joint_values=None, return_all=False):
        if joint_values is None and self.joint_limits is not None:
            joint_values = self.default_joint_values()
        elif joint_values is not None:
            joint_values = self.validate_joint_values(joint_values, check_limits=self.joint_limits is not None)
        return forward_kinematics(self.links, joint_values=joint_values, base=self.base, tool=self.tool, return_all=return_all)

    def forward_kinematics(self, joint_values=None, return_all=False):
        """Canonical long-form alias for :meth:`fk`."""

        return self.fk(joint_values=joint_values, return_all=return_all)

    def positions(self, joint_values=None):
        if joint_values is None and self.joint_limits is not None:
            joint_values = self.default_joint_values()
        elif joint_values is not None:
            joint_values = self.validate_joint_values(joint_values, check_limits=self.joint_limits is not None)
        return chain_positions(self.links, joint_values=joint_values, base=self.base, tool=self.tool)

    def frame_positions(self, joint_values=None):
        """Canonical alias for the chain frame positions."""

        return self.positions(joint_values=joint_values)

    def jacobian(self, joint_values=None):
        if joint_values is None and self.joint_limits is not None:
            joint_values = self.default_joint_values()
        elif joint_values is not None:
            joint_values = self.validate_joint_values(joint_values, check_limits=self.joint_limits is not None)
        return geometric_jacobian(self.links, joint_values=joint_values, base=self.base, tool=self.tool)

    def geometric_jacobian(self, joint_values=None):
        """Canonical long-form alias for :meth:`jacobian`."""

        return self.jacobian(joint_values=joint_values)

    def fk_batch(self, joint_values):
        """Evaluate forward kinematics for a batch of configurations."""

        values = self.validate_joint_values(joint_values, batch=True, check_limits=self.joint_limits is not None)
        return forward_kinematics_batch(self.links, values, base=self.base, tool=self.tool)

    def positions_batch(self, joint_values):
        """Evaluate end-effector positions for a batch of configurations."""

        return self.fk_batch(joint_values)[:, :3, 3]

    def jacobian_batch(self, joint_values):
        """Evaluate geometric Jacobians for a batch of configurations."""

        values = self.validate_joint_values(joint_values, batch=True, check_limits=self.joint_limits is not None)
        return geometric_jacobian_batch(self.links, values, base=self.base, tool=self.tool)

    def end_effector_velocity_batch(self, joint_values, joint_velocities, *, translational=False):
        """Map a batch of joint velocities to end-effector twists."""

        q = self.validate_joint_values(joint_values, batch=True)
        qd = self.validate_joint_values(joint_velocities, batch=True)
        if qd.shape != q.shape:
            raise ValueError("joint_velocities must have the same shape as joint_values")
        jacobian = self.jacobian_batch(q)
        if translational:
            jacobian = jacobian[:, :3, :]
        return np.einsum("bij,bj->bi", jacobian, qd)

    def end_effector_velocity(self, joint_values, joint_velocities, *, translational=False):
        """Map joint velocities to an end-effector twist."""

        q = self.validate_joint_values(joint_values, check_limits=self.joint_limits is not None)
        qd = self.validate_joint_values(joint_velocities)
        jacobian = self.jacobian(q)
        if translational:
            jacobian = jacobian[:3]
        return jacobian @ qd

    def differential_ik(
        self,
        joint_values,
        task_velocity,
        *,
        damping=1e-4,
        translational=False,
    ):
        """Return joint velocities for a desired end-effector velocity.

        A damped least-squares inverse of the geometric Jacobian is used. With
        ``translational=True``, ``task_velocity`` is a 3-vector; otherwise it
        is a 6-vector containing linear and angular velocity.
        """

        q = self.validate_joint_values(joint_values, check_limits=self.joint_limits is not None)
        velocity = np.asarray(task_velocity, dtype=float).reshape(-1)
        expected = 3 if translational else 6
        if velocity.size != expected:
            raise ValueError(f"task_velocity must contain {expected} values")
        if not np.all(np.isfinite(velocity)):
            raise ValueError("task_velocity must contain only finite values")
        damping = float(damping)
        if not np.isfinite(damping) or damping < 0.0:
            raise ValueError("damping must be finite and non-negative")

        jacobian = self.jacobian(q)
        if translational:
            jacobian = jacobian[:3]
        gram = jacobian @ jacobian.T
        regularized = gram + (damping**2) * np.eye(gram.shape[0], dtype=float)
        return jacobian.T @ np.linalg.solve(regularized, velocity)

    def kinematic_metrics(self, joint_values=None, *, translational=False, threshold=1e-8):
        """Return singularity and dexterity diagnostics at a configuration.

        ``translational=True`` evaluates only the linear part of the Jacobian,
        which is usually the useful metric for planar TCP positioning.  The
        returned singular values are ordered from largest to smallest.
        """

        threshold = float(threshold)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("threshold must be a positive finite value")
        jacobian = self.jacobian(joint_values)
        if translational:
            jacobian = jacobian[:3]
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        scale = float(singular_values[0]) if singular_values.size else 0.0
        effective_threshold = threshold * max(1.0, scale)
        rank = int(np.count_nonzero(singular_values > effective_threshold))
        full_rank = min(jacobian.shape)
        smallest = float(singular_values[-1]) if singular_values.size else 0.0
        condition_number = float("inf") if smallest <= effective_threshold else scale / smallest
        return RobotResult({
            "singular_values": singular_values,
            "rank": rank,
            "full_rank": full_rank,
            "singular": rank < full_rank,
            "condition_number": condition_number,
            "manipulability": float(np.prod(singular_values)) if singular_values.size else 0.0,
            "translational": bool(translational),
        })

    def kinematic_metrics_batch(self, joint_values, *, translational=False, threshold=1e-8):
        """Return singularity and dexterity diagnostics for configurations.

        The Jacobian batch uses the selected compiled backend, while the SVD
        reduction stays in NumPy for parity with :meth:`kinematic_metrics`.
        """

        threshold = float(threshold)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("threshold must be a positive finite value")
        values = self.validate_joint_values(
            joint_values, batch=True, check_limits=self.joint_limits is not None
        )
        jacobian = self.jacobian_batch(values)
        if translational:
            jacobian = jacobian[:, :3, :]
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        scale = singular_values[:, 0] if singular_values.shape[1] else np.zeros(values.shape[0])
        effective_threshold = threshold * np.maximum(1.0, scale)
        rank = np.count_nonzero(singular_values > effective_threshold[:, None], axis=1)
        full_rank = min(jacobian.shape[1:])
        smallest = singular_values[:, -1] if singular_values.shape[1] else np.zeros(values.shape[0])
        condition_number = np.divide(
            scale,
            smallest,
            out=np.full(values.shape[0], np.inf, dtype=float),
            where=smallest > effective_threshold,
        )
        return RobotResult({
            "singular_values": singular_values,
            "rank": rank.astype(int),
            "full_rank": full_rank,
            "singular": rank < full_rank,
            "condition_number": condition_number,
            "manipulability": np.prod(singular_values, axis=1)
            if singular_values.shape[1]
            else np.zeros(values.shape[0]),
            "translational": bool(translational),
        })

    def ik(self, target, joint_values=None, **kwargs):
        if joint_values is not None:
            joint_values = self.validate_joint_values(joint_values, check_limits=self.joint_limits is not None)
        kwargs = dict(kwargs)
        kwargs.setdefault("joint_limits", self.joint_limits)
        return inverse_kinematics(target, self.links, joint_values=joint_values, base=self.base, tool=self.tool, **kwargs)

    def inverse_kinematics(self, target, joint_values=None, **kwargs):
        """Canonical long-form alias for :meth:`ik`."""

        return self.ik(target, joint_values=joint_values, **kwargs)

    def ik_batch(self, targets, joint_values=None, *, warm_start=True, **kwargs):
        """Solve IK for a sequence of position or pose targets."""

        kwargs = dict(kwargs)
        kwargs.setdefault("joint_limits", self.joint_limits)
        return inverse_kinematics_batch(
            targets,
            self.links,
            joint_values=joint_values,
            base=self.base,
            tool=self.tool,
            warm_start=warm_start,
            **kwargs,
        )

    def plot(self, joint_values=None, ax=None, annotate=False):
        points = self.positions(joint_values=joint_values)
        return plot_chain(points[:, :2] if points.shape[1] >= 2 else points, ax=ax, annotate=annotate)
