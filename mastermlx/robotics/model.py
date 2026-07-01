from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics import DHLink, chain_positions, forward_kinematics, inverse_kinematics
from .jacobian import geometric_jacobian
from .urdf_parser import parse_urdf, urdf_to_dh_chain
from .visualizer import plot_chain


@dataclass
class RobotModel:
    """Lightweight serial robot model wrapper."""

    links: list[DHLink]
    name: str = "robot"
    base: np.ndarray | None = None
    tool: np.ndarray | None = None

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
        return forward_kinematics(self.links, joint_values=joint_values, base=self.base, tool=self.tool, return_all=return_all)

    def positions(self, joint_values=None):
        return chain_positions(self.links, joint_values=joint_values, base=self.base, tool=self.tool)

    def jacobian(self, joint_values=None):
        return geometric_jacobian(self.links, joint_values=joint_values, base=self.base, tool=self.tool)

    def ik(self, target, joint_values=None, **kwargs):
        return inverse_kinematics(target, self.links, joint_values=joint_values, base=self.base, tool=self.tool, **kwargs)

    def plot(self, joint_values=None, ax=None, annotate=False):
        points = self.positions(joint_values=joint_values)
        return plot_chain(points[:, :2] if points.shape[1] >= 2 else points, ax=ax, annotate=annotate)
