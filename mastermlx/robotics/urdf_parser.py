from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np

from .kinematics import DHLink


@dataclass(frozen=True)
class URDFJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis: tuple[float, float, float]


@dataclass(frozen=True)
class URDFLink:
    name: str


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
        joints.append(
            URDFJoint(
                name=name,
                joint_type=joint_type,
                parent=parent.attrib["link"] if parent is not None else "",
                child=child.attrib["link"] if child is not None else "",
                origin_xyz=_parse_vector(origin.attrib.get("xyz") if origin is not None else None, 3),
                origin_rpy=_parse_vector(origin.attrib.get("rpy") if origin is not None else None, 3),
                axis=_parse_vector(axis.attrib.get("xyz") if axis is not None else None, 3, default=0.0),
            )
        )
    return links, joints


def urdf_to_dh_chain(xml_text, base_link=None, tip_link=None):
    """Convert a simple serial URDF chain into a DHLink list.

    This is intentionally conservative: only serial chains of revolute/prismatic
    joints with pure X/Y/Z axis-aligned origins are mapped automatically.
    """

    links, joints = parse_urdf(xml_text)
    if not joints:
        return []

    if base_link is None:
        base_link = joints[0].parent
    if tip_link is None:
        tip_link = joints[-1].child

    chain = []
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
        current = joint.child
        if current == tip_link:
            break
    return chain
