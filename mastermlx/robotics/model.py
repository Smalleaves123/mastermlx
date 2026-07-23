from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics import DHLink, chain_positions, forward_kinematics, inverse_kinematics
from .jacobian import geometric_jacobian
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

    @property
    def n_joints(self):
        """Number of actuated joints in the serial chain."""

        return len(self.links)

    def validate_joint_values(self, joint_values, *, batch=False):
        """Validate and normalize one or more joint configurations."""

        values = np.asarray(joint_values, dtype=float)
        if batch:
            if values.ndim != 2 or values.shape[1] != self.n_joints:
                raise ValueError(f"joint_values must have shape (n_samples, {self.n_joints})")
        elif values.reshape(-1).size != self.n_joints:
            raise ValueError(f"joint_values must contain {self.n_joints} values")
        if not np.all(np.isfinite(values)):
            raise ValueError("joint_values must contain only finite values")
        return values if batch else values.reshape(self.n_joints)

    @classmethod
    def from_urdf(cls, xml_text, *, name=None, base_link=None, tip_link=None):
        links = urdf_to_dh_chain(xml_text, base_link=base_link, tip_link=tip_link)
        if name is None:
            parsed_links, _ = parse_urdf(xml_text)
            name = "robot" if not parsed_links else parsed_links[0].name
        return cls(links=links, name=name)

    @classmethod
    def from_dh(cls, links, *, name="robot", base=None, tool=None):
        """Build a robot model from DH links or link-like dictionaries."""

        return cls(links=[link if isinstance(link, DHLink) else DHLink(**link) for link in links], name=name, base=base, tool=tool)

    def fk(self, joint_values=None, return_all=False):
        if joint_values is not None:
            joint_values = self.validate_joint_values(joint_values)
        return forward_kinematics(self.links, joint_values=joint_values, base=self.base, tool=self.tool, return_all=return_all)

    def forward_kinematics(self, joint_values=None, return_all=False):
        """Canonical long-form alias for :meth:`fk`."""

        return self.fk(joint_values=joint_values, return_all=return_all)

    def positions(self, joint_values=None):
        if joint_values is not None:
            joint_values = self.validate_joint_values(joint_values)
        return chain_positions(self.links, joint_values=joint_values, base=self.base, tool=self.tool)

    def frame_positions(self, joint_values=None):
        """Canonical alias for the chain frame positions."""

        return self.positions(joint_values=joint_values)

    def jacobian(self, joint_values=None):
        if joint_values is not None:
            joint_values = self.validate_joint_values(joint_values)
        return geometric_jacobian(self.links, joint_values=joint_values, base=self.base, tool=self.tool)

    def geometric_jacobian(self, joint_values=None):
        """Canonical long-form alias for :meth:`jacobian`."""

        return self.jacobian(joint_values=joint_values)

    def fk_batch(self, joint_values):
        """Evaluate forward kinematics for a batch of configurations."""

        values = self.validate_joint_values(joint_values, batch=True)
        return np.asarray([self.fk(q) for q in values], dtype=float)

    def jacobian_batch(self, joint_values):
        """Evaluate geometric Jacobians for a batch of configurations."""

        values = self.validate_joint_values(joint_values, batch=True)
        return np.asarray([self.jacobian(q) for q in values], dtype=float)

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

    def ik(self, target, joint_values=None, **kwargs):
        if joint_values is not None:
            joint_values = self.validate_joint_values(joint_values)
        return inverse_kinematics(target, self.links, joint_values=joint_values, base=self.base, tool=self.tool, **kwargs)

    def inverse_kinematics(self, target, joint_values=None, **kwargs):
        """Canonical long-form alias for :meth:`ik`."""

        return self.ik(target, joint_values=joint_values, **kwargs)

    def plot(self, joint_values=None, ax=None, annotate=False):
        points = self.positions(joint_values=joint_values)
        return plot_chain(points[:, :2] if points.shape[1] >= 2 else points, ax=ax, annotate=annotate)
