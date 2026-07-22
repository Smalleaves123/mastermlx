from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from ..robotics.model import RobotModel
from ..robotics.visualizer import plot_chain
from ..planning import rrt
from .core import SimpleRobotSim


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
        values = tuple(map(float, center))
        if len(values) != 2:
            raise ValueError("obstacle center must contain exactly two coordinates")
        point = (values[0], values[1])
        self.obstacles.append(CircleObstacle(point, float(radius)))
        return self.obstacles[-1]

    def link_positions(self, joint_values=None):
        points = self.robot.positions(joint_values)
        if points.shape[1] >= 2:
            return points[:, :2]
        return points

    @staticmethod
    def _seg_dist(point, start, end):
        edge = end - start
        length_sq = float(np.dot(edge, edge))
        if length_sq == 0.0:
            return float(np.linalg.norm(point - start))
        t = float(np.dot(point - start, edge) / length_sq)
        t = min(1.0, max(0.0, t))
        return float(np.linalg.norm(point - (start + t * edge)))

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
        for idx, (start, end) in enumerate(zip(points[:-1], points[1:])):
            for obstacle in self.obstacles:
                dist = self._seg_dist(
                    np.asarray(obstacle.center, dtype=float),
                    np.asarray(start[:2], dtype=float),
                    np.asarray(end[:2], dtype=float),
                )
                if dist <= obstacle.radius:
                    hits.append(
                        {
                            "segment_index": idx,
                            "obstacle": obstacle,
                            "distance": dist,
                        }
                    )
        return hits

    def hit(self, joint_values=None):
        """Return whether any joint, link segment, or obstacle overlaps."""

        return bool(self.collision_report(joint_values))

    def clearance(self, joint_values=None):
        """Return the smallest planar clearance from robot links to obstacles.

        A positive value means that every joint and link segment is separated
        from every obstacle.  ``np.inf`` is returned when the world contains
        no obstacles.
        """

        if not self.obstacles:
            return float("inf")

        points = self.link_positions(joint_values)
        minimum = float("inf")
        for point in points:
            for obstacle in self.obstacles:
                distance = float(np.linalg.norm(point[:2] - np.asarray(obstacle.center, dtype=float)))
                minimum = min(minimum, distance - obstacle.radius)
        for start, end in zip(points[:-1], points[1:]):
            for obstacle in self.obstacles:
                distance = self._seg_dist(
                    np.asarray(obstacle.center, dtype=float),
                    np.asarray(start[:2], dtype=float),
                    np.asarray(end[:2], dtype=float),
                )
                minimum = min(minimum, distance - obstacle.radius)
        return minimum

    def plan_path(self, q_start, q_goal, bounds, **kwargs):
        """Plan a collision-free path in joint space."""

        return rrt(q_start, q_goal, bounds, hit=self.hit, **kwargs)

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

    def trajectory_follow(self, trajectory, gains=(4.0, 0.4), dt=0.1, damping=0.0, state=None):
        """Track a joint trajectory with a simple PD joint-space controller."""

        trajectory = np.asarray(trajectory, dtype=float)
        if trajectory.ndim != 2 or trajectory.shape[1] != len(self.robot.links):
            raise ValueError("trajectory must have shape (T, n_joints)")
        kp, kd = gains
        sim = SimpleRobotSim(self.robot, state=state, dt=dt, damping=damping)
        states = [sim.state.copy()]
        poses = [sim.pose()]
        controls = []
        for target in trajectory:
            q_err = target - sim.q
            qd_err = -sim.qd
            action = kp * q_err + kd * qd_err
            controls.append(action)
            sim.step(action)
            states.append(sim.state.copy())
            poses.append(sim.pose())
        return np.asarray(states), poses, np.asarray(controls)


def load_world_config(config):
    """Load a simple world configuration from a dict, JSON string, or JSON file."""

    if isinstance(config, (str, Path)):
        try:
            data = json.loads(str(config))
        except json.JSONDecodeError:
            path = Path(config)
            data = json.loads(path.read_text())
    else:
        data = dict(config)

    robot_cfg = data.get("robot", {})
    robot = RobotModel.from_dh(
        robot_cfg.get("links", []),
        name=robot_cfg.get("name", "robot"),
        base=np.asarray(robot_cfg["base"], dtype=float) if "base" in robot_cfg else None,
        tool=np.asarray(robot_cfg["tool"], dtype=float) if "tool" in robot_cfg else None,
    )
    world = SimpleWorld(robot)
    for obstacle in data.get("obstacles", []):
        world.add_obstacle(obstacle["center"], obstacle["radius"])
    state = data.get("state")
    if state is not None:
        state = np.asarray(state, dtype=float).reshape(-1)
    sim_cfg = data.get("sim", {})
    return world, state, sim_cfg
