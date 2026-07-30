"""Collision and clearance helpers for serial robot chains."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from functools import lru_cache
import importlib
from typing import Any

import numpy as np

from ..config import get_backend
from .results import RobotResult


@lru_cache(maxsize=3)
def _load_cpp_collision(backend=None):
    """Load the optional C++ batch clearance kernel for the auto backend."""

    if backend is None:
        backend = get_backend()
    if backend != "auto":
        return None
    try:
        return importlib.import_module("mastermlx.robotics._collision_cpp")
    except ImportError:
        return None


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


def _pack_obstacles(obstacles, point_dims):
    """Pack supported obstacle objects into the fixed C++ array contract."""

    obstacles = list(obstacles)
    types = np.empty(len(obstacles), dtype=np.int8)
    dims = np.empty(len(obstacles), dtype=np.int8)
    params = np.zeros((len(obstacles), 7), dtype=float)
    for index, obstacle in enumerate(obstacles):
        dimension = _obstacle_dims(obstacle)
        if dimension > point_dims or dimension > 3:
            return None
        dims[index] = dimension
        if hasattr(obstacle, "center"):
            types[index] = 0
            center = _point(obstacle.center, dimension, "center")
            radius = float(obstacle.radius)
            if radius < 0.0 or not np.isfinite(radius):
                raise ValueError("radius must be a non-negative finite value")
            params[index, :dimension] = center
            params[index, 3] = radius
        elif hasattr(obstacle, "lower") and hasattr(obstacle, "upper"):
            types[index] = 1
            lower = _point(obstacle.lower, dimension, "lower")
            upper = _point(obstacle.upper, dimension, "upper")
            if np.any(lower >= upper):
                raise ValueError("box lower bounds must be strictly below upper bounds")
            params[index, :dimension] = lower
            params[index, 3 : 3 + dimension] = upper
        else:
            types[index] = 2
            start = _point(obstacle.start, dimension, "start")
            end = _point(obstacle.end, dimension, "end")
            radius = float(obstacle.radius)
            if radius < 0.0 or not np.isfinite(radius):
                raise ValueError("radius must be a non-negative finite value")
            params[index, :dimension] = start
            params[index, 3 : 3 + dimension] = end
            params[index, 6] = radius
    return np.ascontiguousarray(types), np.ascontiguousarray(dims), np.ascontiguousarray(params)


def chain_clearance_batch(points, obstacles, *, link_radius=0.0, box_samples=25):
    """Return minimum signed clearance for a batch of robot chains.

    ``points`` has shape ``(n_samples, n_chain_points, n_dims)``.  The C++
    path handles 1D-3D sphere, box, and capsule obstacles; higher-dimensional
    inputs retain the Python fallback.
    """

    points = np.asarray(points, dtype=float)
    if points.ndim != 3 or points.shape[0] < 1 or points.shape[1] < 1 or points.shape[2] < 1:
        raise ValueError("points must have shape (n_samples, n_points, n_dims)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain only finite values")
    link_radius = float(link_radius)
    if link_radius < 0.0 or not np.isfinite(link_radius):
        raise ValueError("link_radius must be a non-negative finite value")
    box_samples = int(box_samples)
    if box_samples < 2:
        raise ValueError("box_samples must be at least 2")
    obstacles = list(obstacles)
    if not obstacles:
        return np.full(points.shape[0], np.inf, dtype=float)

    packed = _pack_obstacles(obstacles, points.shape[2])
    cpp = _load_cpp_collision(get_backend())
    if packed is not None and cpp is not None and callable(getattr(cpp, "chain_clearance_batch", None)):
        types, dims, params = packed
        return np.asarray(
            cpp.chain_clearance_batch(
                np.ascontiguousarray(points), types, dims, params, link_radius, box_samples
            ),
            dtype=float,
        )
    return np.asarray(
        [chain_collision_report(chain, obstacles, link_radius=link_radius)["minimum_clearance"] for chain in points],
        dtype=float,
    )


def chain_collision_free_batch(
    points, obstacles, *, clearance=0.0, link_radius=0.0, box_samples=25
):
    """Return whether each chain in a batch satisfies a clearance threshold.

    The compiled path uses obstacle and chain AABBs as a conservative
    broad-phase filter, then evaluates exact geometry only for candidates.
    """

    points = np.asarray(points, dtype=float)
    if points.ndim != 3 or points.shape[0] < 1 or points.shape[1] < 1 or points.shape[2] < 1:
        raise ValueError("points must have shape (n_samples, n_points, n_dims)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain only finite values")
    clearance = float(clearance)
    link_radius = float(link_radius)
    if clearance < 0.0 or not np.isfinite(clearance):
        raise ValueError("clearance must be a non-negative finite value")
    if link_radius < 0.0 or not np.isfinite(link_radius):
        raise ValueError("link_radius must be a non-negative finite value")
    box_samples = int(box_samples)
    if box_samples < 2:
        raise ValueError("box_samples must be at least 2")
    obstacles = list(obstacles)
    if not obstacles:
        return np.ones(points.shape[0], dtype=bool)

    packed = _pack_obstacles(obstacles, points.shape[2])
    cpp = _load_cpp_collision(get_backend())
    if packed is not None and cpp is not None and callable(
        getattr(cpp, "chain_collision_free_batch", None)
    ):
        types, dims, params = packed
        return np.asarray(
            cpp.chain_collision_free_batch(
                np.ascontiguousarray(points),
                types,
                dims,
                params,
                clearance,
                link_radius,
                box_samples,
            ),
            dtype=bool,
        )
    return np.asarray(
        [
            chain_collision_report(chain, obstacles, link_radius=link_radius)["minimum_clearance"]
            >= clearance
            for chain in points
        ],
        dtype=bool,
    )


def chain_collision_summary_batch(
    points, obstacles, *, link_radius=0.0, box_samples=25
):
    """Return typed collision summaries for each chain in a batch.

    The closest kind is encoded as ``0`` for none, ``1`` for a point, and
    ``2`` for a segment.  Indices are ``-1`` when no obstacle is present.
    """

    points = np.asarray(points, dtype=float)
    if points.ndim != 3 or points.shape[0] < 1 or points.shape[1] < 1 or points.shape[2] < 1:
        raise ValueError("points must have shape (n_samples, n_points, n_dims)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain only finite values")
    link_radius = float(link_radius)
    if link_radius < 0.0 or not np.isfinite(link_radius):
        raise ValueError("link_radius must be a non-negative finite value")
    box_samples = int(box_samples)
    if box_samples < 2:
        raise ValueError("box_samples must be at least 2")
    obstacles = list(obstacles)
    if not obstacles:
        samples = points.shape[0]
        return RobotResult({
            "minimum_clearance": np.full(samples, np.inf, dtype=float),
            "collision": np.zeros(samples, dtype=bool),
            "closest_kind": np.zeros(samples, dtype=np.int8),
            "closest_index": np.full(samples, -1, dtype=np.int64),
            "closest_obstacle_index": np.full(samples, -1, dtype=np.int64),
        })

    packed = _pack_obstacles(obstacles, points.shape[2])
    cpp = _load_cpp_collision(get_backend())
    if packed is not None and cpp is not None and callable(
        getattr(cpp, "chain_collision_summary_batch", None)
    ):
        types, dims, params = packed
        clearances, collisions, kinds, indices, obstacle_indices = cpp.chain_collision_summary_batch(
            np.ascontiguousarray(points), types, dims, params, link_radius, box_samples
        )
        return RobotResult({
            "minimum_clearance": np.asarray(clearances, dtype=float),
            "collision": np.asarray(collisions, dtype=bool),
            "closest_kind": np.asarray(kinds, dtype=np.int8),
            "closest_index": np.asarray(indices, dtype=np.int64),
            "closest_obstacle_index": np.asarray(obstacle_indices, dtype=np.int64),
        })

    reports = [chain_collision_report(chain, obstacles, link_radius=link_radius) for chain in points]
    kind_codes = {None: 0, "point": 1, "segment": 2}
    return RobotResult({
        "minimum_clearance": np.asarray(
            [report["minimum_clearance"] for report in reports], dtype=float
        ),
        "collision": np.asarray([report["collision"] for report in reports], dtype=bool),
        "closest_kind": np.asarray(
            [kind_codes[report["closest"]["kind"]] for report in reports], dtype=np.int8
        ),
        "closest_index": np.asarray(
            [report["closest"]["index"] if report["closest"]["index"] is not None else -1 for report in reports],
            dtype=np.int64,
        ),
        "closest_obstacle_index": np.asarray(
            [
                report["closest"]["obstacle_index"]
                if report["closest"]["obstacle_index"] is not None
                else -1
                for report in reports
            ],
            dtype=np.int64,
        ),
    })


_COLLISION_DETAILS_KEYS = (
    "minimum_clearance",
    "collision",
    "closest_kind",
    "closest_index",
    "closest_obstacle_index",
    "hit_count",
    "hit_truncated",
    "hit_kind",
    "hit_index",
    "hit_obstacle_index",
    "hit_clearance",
)


def _collision_details_buffers(output, samples, max_hits):
    shapes = {
        "minimum_clearance": ((samples,), float),
        "collision": ((samples,), bool),
        "closest_kind": ((samples,), np.int8),
        "closest_index": ((samples,), np.int64),
        "closest_obstacle_index": ((samples,), np.int64),
        "hit_count": ((samples,), np.int64),
        "hit_truncated": ((samples,), bool),
        "hit_kind": ((samples, max_hits), np.int8),
        "hit_index": ((samples, max_hits), np.int64),
        "hit_obstacle_index": ((samples, max_hits), np.int64),
        "hit_clearance": ((samples, max_hits), float),
    }
    if output is None:
        return {
            key: np.empty(shape, dtype=dtype) for key, (shape, dtype) in shapes.items()
        }
    if not isinstance(output, dict):
        raise ValueError("output must be a mapping of named contiguous NumPy arrays")
    buffers = {}
    for key, (shape, dtype) in shapes.items():
        value = output.get(key)
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype(dtype)
            or not value.flags.c_contiguous
            or value.shape != shape
        ):
            raise ValueError(f"output[{key!r}] must be a contiguous {np.dtype(dtype)} array with shape {shape}")
        buffers[key] = value
    return buffers


def chain_collision_details_batch(
    points, obstacles, *, link_radius=0.0, box_samples=25, max_hits=None, output=None
):
    """Return typed detailed collision data for a batch of chains.

    Hit arrays have shape ``(n_samples, max_hits)`` and store point/segment
    kind codes (1/2), element indices, obstacle indices, and signed
    clearances.  ``hit_count`` is the total number of hits; when the supplied
    capacity is too small, only the prefix is stored and ``hit_truncated`` is
    true.  ``output`` may contain all named arrays to reuse their storage.
    """

    points = np.asarray(points, dtype=float)
    if points.ndim != 3 or points.shape[0] < 1 or points.shape[1] < 1 or points.shape[2] < 1:
        raise ValueError("points must have shape (n_samples, n_points, n_dims)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain only finite values")
    link_radius = float(link_radius)
    if link_radius < 0.0 or not np.isfinite(link_radius):
        raise ValueError("link_radius must be a non-negative finite value")
    box_samples = int(box_samples)
    if box_samples < 2:
        raise ValueError("box_samples must be at least 2")
    obstacles = list(obstacles)
    maximum_hits = len(obstacles) * (2 * points.shape[1] - 1)
    if max_hits is None:
        max_hits = maximum_hits
    max_hits = int(max_hits)
    if max_hits < 0:
        raise ValueError("max_hits must be non-negative")
    buffers = _collision_details_buffers(output, points.shape[0], max_hits)

    packed = _pack_obstacles(obstacles, points.shape[2])
    cpp = _load_cpp_collision(get_backend())
    if packed is not None and cpp is not None and callable(
        getattr(cpp, "chain_collision_details_batch", None)
    ):
        types, dims, params = packed
        values = cpp.chain_collision_details_batch(
            np.ascontiguousarray(points),
            types,
            dims,
            params,
            link_radius,
            box_samples,
            max_hits,
            buffers["minimum_clearance"],
            buffers["collision"],
            buffers["closest_kind"],
            buffers["closest_index"],
            buffers["closest_obstacle_index"],
            buffers["hit_count"],
            buffers["hit_truncated"],
            buffers["hit_kind"],
            buffers["hit_index"],
            buffers["hit_obstacle_index"],
            buffers["hit_clearance"],
        )
        return RobotResult(dict(zip(_COLLISION_DETAILS_KEYS, values)))

    kind_codes = {"point": 1, "segment": 2}
    buffers["hit_kind"].fill(0)
    buffers["hit_index"].fill(-1)
    buffers["hit_obstacle_index"].fill(-1)
    buffers["hit_clearance"].fill(np.inf)
    reports = [chain_collision_report(chain, obstacles, link_radius=link_radius) for chain in points]
    for sample, report in enumerate(reports):
        closest = report["closest"]
        buffers["minimum_clearance"][sample] = report["minimum_clearance"]
        buffers["collision"][sample] = report["collision"]
        buffers["closest_kind"][sample] = 0 if closest["kind"] is None else kind_codes[closest["kind"]]
        buffers["closest_index"][sample] = -1 if closest["index"] is None else closest["index"]
        buffers["closest_obstacle_index"][sample] = (
            -1 if closest["obstacle_index"] is None else closest["obstacle_index"]
        )
        buffers["hit_count"][sample] = len(report["hits"])
        buffers["hit_truncated"][sample] = len(report["hits"]) > max_hits
        for slot, hit in enumerate(report["hits"][:max_hits]):
            buffers["hit_kind"][sample, slot] = kind_codes[hit["kind"]]
            buffers["hit_index"][sample, slot] = (
                hit["point_index"] if hit["kind"] == "point" else hit["segment_index"]
            )
            buffers["hit_obstacle_index"][sample, slot] = hit["obstacle_index"]
            buffers["hit_clearance"][sample, slot] = hit["clearance"]
    return RobotResult(buffers)


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
    hits: list[dict[str, Any]] = []
    closest: dict[str, Any] = {
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
                hit: dict[str, Any] = {
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


def _interpolate_joint_path(path, interpolation_step):
    samples = []
    for index, q in enumerate(path):
        if index:
            previous = path[index - 1]
            count = max(1, int(np.ceil(np.linalg.norm(q - previous) / interpolation_step)))
            for alpha in np.linspace(0.0, 1.0, count + 1)[1:-1]:
                samples.append(previous + alpha * (q - previous))
        samples.append(q)
    return np.asarray(samples, dtype=float)


def path_collision_summary(
    robot,
    joint_path,
    obstacles: Iterable[object],
    *,
    link_radius=0.0,
    interpolation_step=0.05,
):
    """Return batched collision diagnostics along an interpolated joint path.

    This lightweight alternative to :func:`path_collision_report` keeps the
    per-sample results as typed arrays and routes kinematics and collision
    summaries through their optional compiled batch kernels.
    """

    path = np.asarray(joint_path, dtype=float)
    if path.ndim != 2 or path.shape[0] < 1 or path.shape[1] != robot.n_joints:
        raise ValueError("joint_path must have shape (n_points, n_joints)")
    if not np.all(np.isfinite(path)):
        raise ValueError("joint_path must contain only finite values")
    interpolation_step = float(interpolation_step)
    if interpolation_step <= 0.0 or not np.isfinite(interpolation_step):
        raise ValueError("interpolation_step must be a positive finite value")

    samples = _interpolate_joint_path(path, interpolation_step)
    points = robot.frame_positions_batch(samples)
    summary = chain_collision_summary_batch(points, obstacles, link_radius=link_radius)
    clearances = summary["minimum_clearance"]
    collision_indices = np.flatnonzero(summary["collision"])
    finite = clearances[np.isfinite(clearances)]
    return RobotResult({
        "collision": bool(collision_indices.size),
        "minimum_clearance": float("inf") if finite.size == 0 else float(np.min(finite)),
        "first_collision_index": None if collision_indices.size == 0 else int(collision_indices[0]),
        "n_samples": samples.shape[0],
        "samples": samples,
        "clearances": clearances,
        "closest_kind": summary["closest_kind"],
        "closest_index": summary["closest_index"],
        "closest_obstacle_index": summary["closest_obstacle_index"],
    })


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

    obstacles = list(obstacles)
    samples = _interpolate_joint_path(path, interpolation_step)
    reports = []
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
        "n_samples": samples.shape[0],
        "samples": samples,
        "clearances": clearances,
        "reports": reports,
    })


__all__ = [
    "BoxObstacle",
    "CapsuleObstacle",
    "chain_clearance_batch",
    "chain_collision_free_batch",
    "chain_collision_details_batch",
    "chain_collision_summary_batch",
    "SphereObstacle",
    "chain_collision_report",
    "path_collision_summary",
    "path_collision_report",
    "point_obstacle_clearance",
    "point_segment_distance",
    "robot_collision_report",
    "segment_distance",
    "segment_obstacle_clearance",
]
