"""Collision and clearance helpers for serial robot chains."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np

from .results import RobotResult


def _point(values, dims=None, name="point"):
    point = np.asarray(values, dtype=float).reshape(-1)
    if point.size == 0 or (dims is not None and point.size != dims):
        suffix = "" if dims is None else f" with {dims} coordinates"
        raise ValueError(f"{name} must be a non-empty point{suffix}")
    if not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must contain only finite values")
    return point


def _as_points(points, name="points"):
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError(f"{name} must have shape (n_points, n_dims)")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def point_segment_distance(point, start, end):
    """Return the Euclidean distance from a point to a line segment."""

    point = _point(point, name="point")
    start = _point(start, point.size, "start")
    end = _point(end, point.size, "end")
    edge = end - start
    length_sq = float(np.dot(edge, edge))
    if length_sq == 0.0:
        return float(np.linalg.norm(point - start))
    alpha = float(np.dot(point - start, edge) / length_sq)
    alpha = min(1.0, max(0.0, alpha))
    return float(np.linalg.norm(point - (start + alpha * edge)))


def segment_distance(first_start, first_end, second_start, second_end):
    """Return the shortest distance between two 2D or 3D line segments."""

    p1 = _point(first_start, name="first_start")
    q1 = _point(first_end, p1.size, "first_end")
    p2 = _point(second_start, p1.size, "second_start")
    q2 = _point(second_end, p1.size, "second_end")
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))
    eps = 1e-12

    if a <= eps and e <= eps:
        return float(np.linalg.norm(p1 - p2))
    if a <= eps:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = float(np.dot(d1, r))
        if e <= eps:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b
            s = 0.0 if denom <= eps else np.clip((b * f - c * e) / denom, 0.0, 1.0)
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)
    closest_first = p1 + float(s) * d1
    closest_second = p2 + float(t) * d2
    return float(np.linalg.norm(closest_first - closest_second))


@dataclass(frozen=True)
class SphereObstacle:
    """Circular or spherical obstacle depending on center dimensionality."""

    center: tuple[float, ...]
    radius: float

    def __post_init__(self):
        center = _point(self.center, name="center")
        radius = float(self.radius)
        if radius < 0.0 or not np.isfinite(radius):
            raise ValueError("radius must be a non-negative finite value")
        object.__setattr__(self, "center", tuple(float(value) for value in center))
        object.__setattr__(self, "radius", radius)


@dataclass(frozen=True)
class BoxObstacle:
    """Axis-aligned box obstacle."""

    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self):
        lower = _point(self.lower, name="lower")
        upper = _point(self.upper, lower.size, "upper")
        if np.any(lower >= upper):
            raise ValueError("box lower bounds must be strictly below upper bounds")
        object.__setattr__(self, "lower", tuple(float(value) for value in lower))
        object.__setattr__(self, "upper", tuple(float(value) for value in upper))


@dataclass(frozen=True)
class CapsuleObstacle:
    """Capsule obstacle around a line segment."""

    start: tuple[float, ...]
    end: tuple[float, ...]
    radius: float

    def __post_init__(self):
        start = _point(self.start, name="start")
        end = _point(self.end, start.size, "end")
        radius = float(self.radius)
        if radius < 0.0 or not np.isfinite(radius):
            raise ValueError("radius must be a non-negative finite value")
        object.__setattr__(self, "start", tuple(float(value) for value in start))
        object.__setattr__(self, "end", tuple(float(value) for value in end))
        object.__setattr__(self, "radius", radius)


def _obstacle_dims(obstacle):
    if hasattr(obstacle, "center"):
        return len(tuple(obstacle.center))
    if hasattr(obstacle, "lower") and hasattr(obstacle, "upper"):
        return len(tuple(obstacle.lower))
    if hasattr(obstacle, "start") and hasattr(obstacle, "end"):
        return len(tuple(obstacle.start))
    raise TypeError("obstacle must define center/radius, lower/upper, or start/end/radius")


def point_obstacle_clearance(point, obstacle):
    """Return signed point clearance to an obstacle.

    Positive values are separated, zero touches, and negative values overlap.
    """

    dims = _obstacle_dims(obstacle)
    point = _point(point, dims, "point")
    if hasattr(obstacle, "center"):
        center = _point(obstacle.center, dims, "center")
        return float(np.linalg.norm(point - center) - float(obstacle.radius))
    if hasattr(obstacle, "lower") and hasattr(obstacle, "upper"):
        lower = _point(obstacle.lower, dims, "lower")
        upper = _point(obstacle.upper, dims, "upper")
        outside = np.maximum(np.maximum(lower - point, point - upper), 0.0)
        outside_distance = float(np.linalg.norm(outside))
        if outside_distance > 0.0:
            return outside_distance
        return -float(np.min(np.minimum(point - lower, upper - point)))
    start = _point(obstacle.start, dims, "start")
    end = _point(obstacle.end, dims, "end")
    return point_segment_distance(point, start, end) - float(obstacle.radius)


def segment_obstacle_clearance(start, end, obstacle, *, box_samples=25):
    """Return signed segment clearance to an obstacle."""

    dims = _obstacle_dims(obstacle)
    start = _point(start, dims, "start")
    end = _point(end, dims, "end")
    if hasattr(obstacle, "center"):
        center = _point(obstacle.center, dims, "center")
        return point_segment_distance(center, start, end) - float(obstacle.radius)
    if hasattr(obstacle, "lower") and hasattr(obstacle, "upper"):
        box_samples = int(box_samples)
        if box_samples < 2:
            raise ValueError("box_samples must be at least 2")
        clearances = [
            point_obstacle_clearance(start + alpha * (end - start), obstacle)
            for alpha in np.linspace(0.0, 1.0, box_samples)
        ]
        return float(np.min(clearances))
    return segment_distance(start, end, obstacle.start, obstacle.end) - float(obstacle.radius)


def chain_collision_report(points, obstacles, *, link_radius=0.0):
    """Summarize clearance between a kinematic chain and obstacles."""

    points = _as_points(points)
    obstacles = list(obstacles)
    link_radius = float(link_radius)
    if link_radius < 0.0 or not np.isfinite(link_radius):
        raise ValueError("link_radius must be a non-negative finite value")
    hits = []
    closest = {
        "kind": None,
        "index": None,
        "obstacle_index": None,
        "clearance": float("inf"),
    }
    if not obstacles:
        return RobotResult({
            "collision": False,
            "minimum_clearance": float("inf"),
            "closest": closest,
            "hits": hits,
        })

    for obstacle_index, obstacle in enumerate(obstacles):
        dims = _obstacle_dims(obstacle)
        chain_points = points[:, :dims]
        for point_index, point in enumerate(chain_points):
            clearance = point_obstacle_clearance(point, obstacle) - link_radius
            if clearance < closest["clearance"]:
                closest = {
                    "kind": "point",
                    "index": point_index,
                    "obstacle_index": obstacle_index,
                    "clearance": float(clearance),
                }
            if clearance <= 0.0:
                hit = {
                    "kind": "point",
                    "point_index": point_index,
                    "obstacle_index": obstacle_index,
                    "clearance": float(clearance),
                    "obstacle": obstacle,
                }
                if hasattr(obstacle, "radius"):
                    hit["distance"] = float(clearance + link_radius + float(obstacle.radius))
                hits.append(hit)
        for segment_index, (start, end) in enumerate(zip(chain_points[:-1], chain_points[1:])):
            clearance = segment_obstacle_clearance(start, end, obstacle) - link_radius
            if clearance < closest["clearance"]:
                closest = {
                    "kind": "segment",
                    "index": segment_index,
                    "obstacle_index": obstacle_index,
                    "clearance": float(clearance),
                }
            if clearance <= 0.0:
                hit = {
                    "kind": "segment",
                    "segment_index": segment_index,
                    "obstacle_index": obstacle_index,
                    "clearance": float(clearance),
                    "obstacle": obstacle,
                }
                if hasattr(obstacle, "radius"):
                    hit["distance"] = float(clearance + link_radius + float(obstacle.radius))
                hits.append(hit)
    return RobotResult({
        "collision": bool(hits),
        "minimum_clearance": float(closest["clearance"]),
        "closest": closest,
        "hits": hits,
    })


def robot_collision_report(robot, joint_values=None, obstacles: Iterable[object] = (), *, link_radius=0.0):
    """Return collision diagnostics for a robot model at one configuration."""

    return chain_collision_report(robot.positions(joint_values), obstacles, link_radius=link_radius)


def path_collision_report(
    robot,
    joint_path,
    obstacles: Iterable[object],
    *,
    link_radius=0.0,
    interpolation_step=0.05,
):
    """Return collision and clearance diagnostics along a joint-space path."""

    path = np.asarray(joint_path, dtype=float)
    if path.ndim != 2 or path.shape[0] < 1 or path.shape[1] != robot.n_joints:
        raise ValueError("joint_path must have shape (n_points, n_joints)")
    if not np.all(np.isfinite(path)):
        raise ValueError("joint_path must contain only finite values")
    interpolation_step = float(interpolation_step)
    if interpolation_step <= 0.0 or not np.isfinite(interpolation_step):
        raise ValueError("interpolation_step must be a positive finite value")

    samples = []
    reports = []
    for index, q in enumerate(path):
        if index:
            previous = path[index - 1]
            count = max(1, int(np.ceil(np.linalg.norm(q - previous) / interpolation_step)))
            for alpha in np.linspace(0.0, 1.0, count + 1)[1:-1]:
                samples.append(previous + alpha * (q - previous))
        samples.append(q)
    for q in samples:
        reports.append(robot_collision_report(robot, q, obstacles, link_radius=link_radius))
    clearances = np.asarray([report["minimum_clearance"] for report in reports], dtype=float)
    finite = clearances[np.isfinite(clearances)]
    minimum = float("inf") if finite.size == 0 else float(np.min(finite))
    first_collision = next((idx for idx, report in enumerate(reports) if report["collision"]), None)
    return RobotResult({
        "collision": any(report["collision"] for report in reports),
        "minimum_clearance": minimum,
        "first_collision_index": first_collision,
        "n_samples": len(samples),
        "samples": np.asarray(samples, dtype=float),
        "clearances": clearances,
        "reports": reports,
    })


__all__ = [
    "BoxObstacle",
    "CapsuleObstacle",
    "SphereObstacle",
    "chain_collision_report",
    "path_collision_report",
    "point_obstacle_clearance",
    "point_segment_distance",
    "robot_collision_report",
    "segment_distance",
    "segment_obstacle_clearance",
]
