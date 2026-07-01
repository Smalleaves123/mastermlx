from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..robotics.model import RobotModel
from ..robotics.visualizer import plot_chain


@dataclass(frozen=True)
class CircleObstacle:
    center: tuple[float, float]
    radius: float


@dataclass
class SimpleWorld:
    """Minimal 2D world containing one robot and circular obstacles."""

    robot: RobotModel
    obstacles: list[CircleObstacle] = field(default_factory=list)

    def add_obstacle(self, center, radius):
        self.obstacles.append(CircleObstacle(tuple(map(float, center)), float(radius)))
        return self.obstacles[-1]

    def link_positions(self, joint_values=None):
        points = self.robot.positions(joint_values)
        if points.shape[1] >= 2:
            return points[:, :2]
        return points

    def collision_report(self, joint_values=None):
        points = self.link_positions(joint_values)
        hits = []
        for idx, point in enumerate(points):
            for obstacle in self.obstacles:
                dist = float(np.linalg.norm(point[:2] - np.asarray(obstacle.center, dtype=float)))
                if dist <= obstacle.radius:
                    hits.append(
                        {
                            "point_index": idx,
                            "obstacle": obstacle,
                            "distance": dist,
                        }
                    )
        return hits

    def lidar_scan(self, joint_values=None, num_rays=64, max_range=10.0):
        """Very small planar range scan against circular obstacles."""

        origin = self.link_positions(joint_values)[-1]
        angles = np.linspace(-np.pi, np.pi, int(num_rays), endpoint=False)
        ranges = np.full_like(angles, float(max_range), dtype=float)
        origin = np.asarray(origin, dtype=float)
        for i, angle in enumerate(angles):
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)
            for obstacle in self.obstacles:
                c = np.asarray(obstacle.center, dtype=float)
                oc = origin - c
                b = 2.0 * np.dot(direction, oc)
                c_term = np.dot(oc, oc) - obstacle.radius**2
                disc = b * b - 4.0 * c_term
                if disc < 0.0:
                    continue
                t = (-b - np.sqrt(disc)) / 2.0
                if 0.0 <= t < ranges[i]:
                    ranges[i] = t
        return angles, ranges

    def render(self, joint_values=None, ax=None, annotate=False):
        points = self.link_positions(joint_values)
        ax = plot_chain(points, ax=ax, annotate=annotate)
        if points.shape[1] == 2:
            import matplotlib.pyplot as plt
            for obstacle in self.obstacles:
                circle = plt.Circle(obstacle.center, obstacle.radius, fill=False, linestyle="--")
                ax.add_patch(circle)
        return ax
