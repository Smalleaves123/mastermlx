from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import TypeAlias

import numpy as np

from ..robotics.collision import (
    BoxObstacle,
    CapsuleObstacle,
    SphereObstacle,
    chain_collision_report,
    point_segment_distance,
)
from ..robotics.model import RobotModel
from ..robotics.visualizer import plot_chain
from ..planning import rrt
from .core import SimpleRobotSim


@dataclass(frozen=True)
class CircleObstacle:
    center: tuple[float, float]
    radius: float


Obstacle: TypeAlias = CircleObstacle | SphereObstacle | BoxObstacle | CapsuleObstacle


def _position3(value, name="position"):
    position = np.asarray(value, dtype=float).reshape(-1)
    if position.size == 2:
        position = np.concatenate([position, [0.0]])
    if position.size != 3 or not np.all(np.isfinite(position)):
        raise ValueError(f"{name} must be a finite 2D or 3D position")
    return position


@dataclass
class SimulationObject:
    """Lightweight movable object used by task-level simulation."""

    name: str
    position: np.ndarray
    radius: float = 0.05
    graspable: bool = True
    attached: bool = field(default=False, init=False)
    _initial_position: np.ndarray = field(default=None, init=False, repr=False)
    _attachment_offset: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.name = str(self.name)
        self.position = _position3(self.position)
        self.radius = float(self.radius)
        if not self.name:
            raise ValueError("object name must not be empty")
        if self.radius <= 0.0 or not np.isfinite(self.radius):
            raise ValueError("object radius must be positive and finite")
        self._initial_position = self.position.copy()
        self._attachment_offset = np.zeros(3, dtype=float)

    def reset(self):
        self.position = self._initial_position.copy()
        self.attached = False
        self._attachment_offset.fill(0.0)

    def attach(self, tcp_position):
        self._attachment_offset = self.position - _position3(tcp_position, "tcp_position")
        self.attached = True

    def sync(self, tcp_position):
        if self.attached:
            self.position = _position3(tcp_position, "tcp_position") + self._attachment_offset

    def detach(self):
        was_attached = self.attached
        self.attached = False
        self._attachment_offset.fill(0.0)
        return was_attached


@dataclass
class SimpleWorld:
    """Minimal 2D world containing one robot and circular obstacles."""

    robot: RobotModel
    obstacles: list[Obstacle] = field(default_factory=list)
    objects: list[SimulationObject] = field(default_factory=list)

    def add_object(self, name, position, radius=0.05, graspable=True):
        """Add a movable object with a canonical three-dimensional position."""

        if any(item.name == str(name) for item in self.objects):
            raise ValueError(f"simulation object {name!r} already exists")
        item = SimulationObject(name, position, radius=radius, graspable=graspable)
        self.objects.append(item)
        return item

    def reset_objects(self):
        for item in self.objects:
            item.reset()

    def attached_object_name(self):
        for item in self.objects:
            if item.attached:
                return item.name
        return None

    def grasp_object(self, tcp_position, max_distance=0.1):
        """Attach the nearest graspable object within ``max_distance``."""

        tcp_position = _position3(tcp_position, "tcp_position")
        max_distance = float(max_distance)
        candidates = [
            item for item in self.objects
            if item.graspable and not item.attached
            and np.linalg.norm(item.position - tcp_position) <= max_distance
        ]
        if not candidates:
            return None
        item = min(candidates, key=lambda value: np.linalg.norm(value.position - tcp_position))
        item.attach(tcp_position)
        return {"command": "grasp", "object": item.name}

    def release_object(self):
        """Release the currently attached object, if any."""

        for item in self.objects:
            if item.attached:
                item.detach()
                return {"command": "release", "object": item.name}
        return None

    def sync_attached_objects(self, tcp_position):
        for item in self.objects:
            item.sync(tcp_position)

    def add_obstacle(self, center, radius):
        values = tuple(map(float, center))
        if len(values) != 2:
            raise ValueError("obstacle center must contain exactly two coordinates")
        point = (values[0], values[1])
        self.obstacles.append(CircleObstacle(point, float(radius)))
        return self.obstacles[-1]

    def add_sphere(self, center, radius):
        """Add a circular or spherical obstacle."""

        obstacle = SphereObstacle(tuple(map(float, center)), float(radius))
        self.obstacles.append(obstacle)
        return obstacle

    def add_box(self, lower, upper):
        """Add an axis-aligned box obstacle."""

        obstacle = BoxObstacle(tuple(map(float, lower)), tuple(map(float, upper)))
        self.obstacles.append(obstacle)
        return obstacle

    def add_capsule(self, start, end, radius):
        """Add a capsule obstacle around a line segment."""

        obstacle = CapsuleObstacle(tuple(map(float, start)), tuple(map(float, end)), float(radius))
        self.obstacles.append(obstacle)
        return obstacle

    def link_positions(self, joint_values=None):
        points = self.robot.positions(joint_values)
        if points.shape[1] >= 2:
            return points[:, :2]
        return points

    @staticmethod
    def _seg_dist(point, start, end):
        return point_segment_distance(point, start, end)

    def collision_report(self, joint_values=None):
        return chain_collision_report(self.link_positions(joint_values), self.obstacles)["hits"]

    def hit(self, joint_values=None):
        """Return whether any joint, link segment, or obstacle overlaps."""

        return bool(self.collision_report(joint_values))

    def clearance(self, joint_values=None):
        """Return the smallest planar clearance from robot links to obstacles.

        A positive value means that every joint and link segment is separated
        from every obstacle.  ``np.inf`` is returned when the world contains
        no obstacles.
        """

        return chain_collision_report(self.link_positions(joint_values), self.obstacles)[
            "minimum_clearance"
        ]

    def plan_path(self, q_start, q_goal, bounds, *, hit=None, **kwargs):
        """Plan a collision-free path in joint space.

        Callers may provide a stricter state predicate, such as a minimum
        clearance requirement.  The default remains the world's collision
        predicate.
        """

        return rrt(q_start, q_goal, bounds, hit=self.hit if hit is None else hit, **kwargs)

    def lidar_scan(self, joint_values=None, num_rays=64, max_range=10.0):
        """Very small planar range scan against circular obstacles."""

        origin = self.link_positions(joint_values)[-1]
        angles = np.linspace(-np.pi, np.pi, int(num_rays), endpoint=False)
        ranges = np.full_like(angles, float(max_range), dtype=float)
        origin = np.asarray(origin, dtype=float)
        for i, angle in enumerate(angles):
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)
            for obstacle in self.obstacles:
                if not isinstance(obstacle, (CircleObstacle, SphereObstacle)):
                    continue
                if len(obstacle.center) != 2:
                    continue
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
                if isinstance(obstacle, (CircleObstacle, SphereObstacle)) and len(obstacle.center) == 2:
                    circle = plt.Circle(obstacle.center, obstacle.radius, fill=False, linestyle="--")
                    ax.add_patch(circle)
                elif isinstance(obstacle, BoxObstacle) and len(obstacle.lower) == 2:
                    lower = np.asarray(obstacle.lower, dtype=float)
                    upper = np.asarray(obstacle.upper, dtype=float)
                    rect = plt.Rectangle(
                        (float(lower[0]), float(lower[1])),
                        float(upper[0] - lower[0]),
                        float(upper[1] - lower[1]),
                        fill=False,
                        linestyle="--",
                    )
                    ax.add_patch(rect)
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
        kind = str(obstacle.get("kind", "circle")).lower()
        if kind in {"circle", "disc", "disk"}:
            world.add_obstacle(obstacle["center"], obstacle["radius"])
        elif kind == "sphere":
            world.add_sphere(obstacle["center"], obstacle["radius"])
        elif kind == "box":
            world.add_box(obstacle["lower"], obstacle["upper"])
        elif kind == "capsule":
            world.add_capsule(obstacle["start"], obstacle["end"], obstacle["radius"])
        else:
            raise ValueError("obstacle kind must be one of: circle, sphere, box, capsule")
    for item in data.get("objects", []):
        world.add_object(
            item["name"],
            item["position"],
            radius=item.get("radius", 0.05),
            graspable=item.get("graspable", True),
        )
    state = data.get("state")
    if state is not None:
        state = np.asarray(state, dtype=float).reshape(-1)
    sim_cfg = data.get("sim", {})
    return world, state, sim_cfg
