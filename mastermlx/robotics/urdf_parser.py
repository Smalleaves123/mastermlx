from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np

from .kinematics import DHLink
from .transforms import homogeneous_transform, rpy_to_matrix


@dataclass(frozen=True)
class URDFJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    limits: tuple[float, float] | None = None


@dataclass(frozen=True)
class URDFLink:
    name: str


def _normalize_joint_axis(axis):
    values = np.asarray(axis, dtype=float).reshape(-1)
    if values.size != 3 or not np.all(np.isfinite(values)):
        raise ValueError("URDF joint axis must be a finite 3-vector")
    norm = float(np.linalg.norm(values))
    if norm <= 1e-12:
        raise ValueError("URDF movable joint axis must be non-zero")
    return values / norm


def _axis_angle_matrix(axis, angle):
    axis = _normalize_joint_axis(axis)
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)
    c = np.cos(float(angle))
    s = np.sin(float(angle))
    return np.eye(3, dtype=float) * c + (1.0 - c) * np.outer(axis, axis) + s * K


def _find_serial_joint_path(joints, links, base_link=None, tip_link=None):
    """Find one connected URDF joint path without assuming XML order."""

    link_names = {link.name for link in links}
    if not joints:
        return []
    if base_link is None:
        roots = link_names - {joint.child for joint in joints}
        base_link = next(iter(roots), joints[0].parent) if len(roots) == 1 else joints[0].parent
    if base_link not in link_names:
        raise ValueError(f"Unknown URDF base_link: {base_link!r}")

    outgoing: dict[str, list[URDFJoint]] = {}
    for joint in joints:
        outgoing.setdefault(joint.parent, []).append(joint)

    if tip_link is None:
        reachable = set()
        stack = [base_link]
        while stack:
            current = stack.pop()
            if current in reachable:
                continue
            reachable.add(current)
            stack.extend(joint.child for joint in outgoing.get(current, ()))
        leaves = [name for name in reachable if not outgoing.get(name)]
        if len(leaves) != 1:
            raise ValueError("tip_link is required for branching or incomplete URDF chains")
        tip_link = leaves[0]
    if tip_link not in link_names:
        raise ValueError(f"Unknown URDF tip_link: {tip_link!r}")

    def search(current, visited):
        if current == tip_link:
            return []
        if current in visited:
            raise ValueError("URDF joint graph contains a cycle")
        for joint in outgoing.get(current, ()):
            suffix = search(joint.child, visited | {current})
            if suffix is not None:
                return [joint, *suffix]
        return None

    path = search(base_link, set())
    if path is None:
        raise ValueError(f"No URDF joint path from {base_link!r} to {tip_link!r}")
    unsupported = {"floating", "planar", "spherical"}
    for joint in path:
        if joint.joint_type not in {"fixed", "revolute", "continuous", "prismatic"}:
            if joint.joint_type in unsupported:
                raise ValueError(f"URDF joint type {joint.joint_type!r} is not supported")
            raise ValueError(f"Unknown URDF joint type: {joint.joint_type!r}")
    return path


@dataclass(frozen=True)
class URDFSerialChain:
    """General serial URDF chain for spatial forward kinematics.

    Unlike :func:`urdf_to_dh_chain`, this representation preserves each joint's
    origin RPY and arbitrary axis. It is intentionally kinematics-only; the
    existing DH model remains the compatibility path for dynamics and legacy
    workcell APIs.
    """

    base_link: str
    tip_link: str
    joints: tuple[URDFJoint, ...]

    @classmethod
    def from_urdf(cls, xml_text, *, base_link=None, tip_link=None):
        links, joints = parse_urdf(xml_text)
        path = _find_serial_joint_path(joints, links, base_link=base_link, tip_link=tip_link)
        if not path:
            raise ValueError("URDF must contain at least one joint for a serial chain")
        base = path[0].parent if base_link is None else base_link
        tip = path[-1].child if tip_link is None else tip_link
        for joint in path:
            if joint.joint_type in {"revolute", "continuous", "prismatic"}:
                _normalize_joint_axis(joint.axis)
        return cls(base_link=base, tip_link=tip, joints=tuple(path))

    @property
    def active_joints(self):
        return tuple(joint for joint in self.joints if joint.joint_type != "fixed")

    @property
    def joint_names(self):
        return tuple(joint.name for joint in self.active_joints)

    @property
    def joint_types(self):
        return tuple(joint.joint_type for joint in self.active_joints)

    @property
    def n_joints(self):
        return len(self.active_joints)

    @property
    def joint_limits(self):
        active = self.active_joints
        if not active or any(joint.limits is None for joint in active):
            return None
        return np.asarray([joint.limits for joint in active], dtype=float)

    def validate_joint_values(self, joint_values=None):
        if joint_values is None:
            values = np.zeros(self.n_joints, dtype=float)
        else:
            values = np.asarray(joint_values, dtype=float).reshape(-1)
        if values.size != self.n_joints:
            raise ValueError(f"joint_values must contain {self.n_joints} values")
        if not np.all(np.isfinite(values)):
            raise ValueError("joint_values must contain only finite values")
        return values

    def _forward_with_geometry(self, joint_values=None, base=None, tool=None):
        q = self.validate_joint_values(joint_values)
        T = np.eye(4, dtype=float) if base is None else np.asarray(base, dtype=float)
        if T.shape != (4, 4):
            raise ValueError("base must have shape (4, 4)")
        frames = [T.copy()]
        origins = []
        axes = []
        q_index = 0
        for joint in self.joints:
            origin = homogeneous_transform(
                rpy_to_matrix(*joint.origin_rpy), joint.origin_xyz
            )
            joint_frame = T @ origin
            if joint.joint_type == "fixed":
                motion = np.eye(4, dtype=float)
            else:
                axis = _normalize_joint_axis(joint.axis)
                origins.append(joint_frame[:3, 3].copy())
                axes.append(joint_frame[:3, :3] @ axis)
                value = q[q_index]
                q_index += 1
                if joint.joint_type in {"revolute", "continuous"}:
                    motion = homogeneous_transform(_axis_angle_matrix(axis, value), [0.0, 0.0, 0.0])
                else:
                    motion = homogeneous_transform(np.eye(3), axis * value)
            T = joint_frame @ motion
            frames.append(T.copy())
        if tool is not None:
            tool = np.asarray(tool, dtype=float)
            if tool.shape != (4, 4):
                raise ValueError("tool must have shape (4, 4)")
            T = T @ tool
            frames.append(T.copy())
        return T, frames, np.asarray(origins), np.asarray(axes)

    def forward_kinematics(self, joint_values=None, *, base=None, tool=None, return_all=False):
        T, frames, _, _ = self._forward_with_geometry(
            joint_values=joint_values, base=base, tool=tool
        )
        return (T, frames) if return_all else T

    def forward_kinematics_batch(self, joint_values, *, base=None, tool=None):
        values = np.asarray(joint_values, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_joints:
            raise ValueError(f"joint_values must have shape (n_samples, {self.n_joints})")
        return np.asarray(
            [self.forward_kinematics(row, base=base, tool=tool) for row in values], dtype=float
        )

    def positions(self, joint_values=None, *, base=None, tool=None):
        _, frames = self.forward_kinematics(
            joint_values, base=base, tool=tool, return_all=True
        )
        return np.asarray([frame[:3, 3] for frame in frames], dtype=float)

    def geometric_jacobian(self, joint_values=None, *, base=None, tool=None):
        T, _, origins, axes = self._forward_with_geometry(
            joint_values=joint_values, base=base, tool=tool
        )
        J = np.zeros((6, self.n_joints), dtype=float)
        p_end = T[:3, 3]
        for index, joint in enumerate(self.active_joints):
            if joint.joint_type in {"revolute", "continuous"}:
                J[:3, index] = np.cross(axes[index], p_end - origins[index])
                J[3:, index] = axes[index]
            else:
                J[:3, index] = axes[index]
        return J

    def geometric_jacobian_batch(self, joint_values, *, base=None, tool=None):
        values = np.asarray(joint_values, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_joints:
            raise ValueError(f"joint_values must have shape (n_samples, {self.n_joints})")
        return np.asarray(
            [self.geometric_jacobian(row, base=base, tool=tool) for row in values], dtype=float
        )


def _parse_vector(text, length, default=0.0):
    if text is None:
        return tuple(float(default) for _ in range(length))
    values = [float(x) for x in text.split()]
    if len(values) != length:
        raise ValueError(f"Expected {length} values, got {len(values)}")
    return tuple(values)


def parse_urdf(xml_text):
    """Parse a minimal URDF model into links and joints."""

    root = ET.fromstring(xml_text)
    if root.tag != "robot":
        raise ValueError("URDF must have a <robot> root element")

    links = [URDFLink(name=node.attrib["name"]) for node in root.findall("link")]
    joints = []
    for node in root.findall("joint"):
        name = node.attrib["name"]
        joint_type = node.attrib.get("type", "fixed")
        parent = node.find("parent")
        child = node.find("child")
        origin = node.find("origin")
        axis = node.find("axis")
        limit = node.find("limit")
        limits = None
        if limit is not None and "lower" in limit.attrib and "upper" in limit.attrib:
            limits = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
        joints.append(
            URDFJoint(
                name=name,
                joint_type=joint_type,
                parent=parent.attrib["link"] if parent is not None else "",
                child=child.attrib["link"] if child is not None else "",
                origin_xyz=_parse_vector(origin.attrib.get("xyz") if origin is not None else None, 3),
                origin_rpy=_parse_vector(origin.attrib.get("rpy") if origin is not None else None, 3),
                axis=_parse_vector(axis.attrib.get("xyz") if axis is not None else None, 3, default=0.0),
                limits=limits,
            )
        )
    return links, joints


def urdf_to_dh_chain(xml_text, base_link=None, tip_link=None, *, return_limits=False):
    """Convert a simple serial URDF chain into a DHLink list.

    This is intentionally conservative: only serial chains of revolute/prismatic
    joints with pure X/Y/Z axis-aligned origins are mapped automatically.
    """

    links, joints = parse_urdf(xml_text)
    if not joints:
        return ([], None) if return_limits else []

    if base_link is None:
        base_link = joints[0].parent
    if tip_link is None:
        tip_link = joints[-1].child

    chain = []
    limits = []
    current = base_link
    for joint in joints:
        if joint.parent != current:
            continue
        if joint.joint_type not in {"revolute", "prismatic"}:
            continue
        xyz = np.asarray(joint.origin_xyz, dtype=float)
        rpy = np.asarray(joint.origin_rpy, dtype=float)
        if np.linalg.norm(rpy) > 1e-12:
            raise ValueError("Only zero-rpy joints are supported by urdf_to_dh_chain")
        if np.count_nonzero(np.abs(xyz) > 1e-12) > 1:
            raise ValueError("Only axis-aligned joint origins are supported by urdf_to_dh_chain")
        a = float(xyz[0])
        d = float(xyz[2])
        theta = 0.0
        alpha = 0.0
        if np.isclose(xyz[1], 0.0):
            pass
        if joint.joint_type == "prismatic":
            chain.append(DHLink(a=a, alpha=alpha, d=d, theta=theta, joint_type="prismatic", offset=0.0))
        else:
            chain.append(DHLink(a=a, alpha=alpha, d=d, theta=theta, joint_type="revolute", offset=0.0))
        limits.append(joint.limits)
        current = joint.child
        if current == tip_link:
            break
    if return_limits:
        return chain, np.asarray(limits, dtype=float) if limits and all(limit is not None for limit in limits) else None
    return chain
