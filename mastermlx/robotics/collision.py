"""Collision and clearance helpers for serial robot chains."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from functools import lru_cache
import importlib
from pathlib import Path
import struct
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


@dataclass(frozen=True)
class MeshObstacle:
    """Triangular mesh obstacle with optional OBJ/STL file loaders."""

    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self):
        vertices = np.asarray(self.vertices, dtype=float)
        faces = np.asarray(self.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 3:
            raise ValueError("mesh vertices must have shape (n_vertices, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] < 1:
            raise ValueError("mesh faces must have shape (n_faces, 3)")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("mesh vertices must contain only finite values")
        if np.any(faces < 0) or np.any(faces >= vertices.shape[0]):
            raise ValueError("mesh faces contain an invalid vertex index")
        object.__setattr__(self, "vertices", np.ascontiguousarray(vertices))
        object.__setattr__(self, "faces", np.ascontiguousarray(faces))

    @property
    def triangles(self):
        """Return mesh triangles with shape ``(n_faces, 3, 3)``."""

        return self.vertices[self.faces]

    @property
    def lower(self):
        return np.min(self.vertices, axis=0)

    @property
    def upper(self):
        return np.max(self.vertices, axis=0)

    def transformed(self, transform):
        """Return this mesh transformed by a homogeneous matrix."""

        transform = np.asarray(transform, dtype=float)
        if transform.shape != (4, 4):
            raise ValueError("transform must have shape (4, 4)")
        homogeneous = np.column_stack([self.vertices, np.ones(self.vertices.shape[0])])
        return MeshObstacle((homogeneous @ transform.T)[:, :3], self.faces)

    @classmethod
    def from_obj(cls, path):
        """Load a triangular or polygonal Wavefront OBJ mesh."""

        vertices: list[list[float]] = []
        faces: list[list[int]] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            fields = line.strip().split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "v" and len(fields) >= 4:
                vertices.append([float(value) for value in fields[1:4]])
            elif fields[0] == "f" and len(fields) >= 4:
                indices = []
                for field in fields[1:]:
                    index = int(field.split("/")[0])
                    indices.append(index - 1 if index > 0 else len(vertices) + index)
                for index in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[index], indices[index + 1]])
        return cls(np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64))

    @classmethod
    def from_stl(cls, path):
        """Load either binary or ASCII STL geometry."""

        raw = Path(path).read_bytes()
        binary_count = struct.unpack_from("<I", raw, 80)[0] if len(raw) >= 84 else -1
        if binary_count >= 0 and 84 + 50 * binary_count == len(raw):
            vertices: list[list[float]] = []
            faces: list[list[int]] = []
            for index in range(binary_count):
                offset = 84 + 50 * index + 12
                triangle = np.frombuffer(raw, dtype="<f4", count=9, offset=offset).astype(float)
                start = len(vertices)
                vertices.extend(triangle.reshape(3, 3).tolist())
                faces.append([start, start + 1, start + 2])
            return cls(np.asarray(vertices), np.asarray(faces, dtype=np.int64))
        ascii_vertices: list[list[float]] = []
        for line in raw.decode("utf-8", errors="ignore").splitlines():
            fields = line.strip().split()
            if len(fields) == 4 and fields[0].lower() == "vertex":
                ascii_vertices.append([float(value) for value in fields[1:4]])
        if len(ascii_vertices) % 3 != 0:
            raise ValueError("ASCII STL must contain complete triangles")
        face_array = np.arange(len(ascii_vertices), dtype=np.int64).reshape(-1, 3)
        return cls(np.asarray(ascii_vertices), face_array)

    @classmethod
    def from_file(cls, path):
        """Load an OBJ or STL mesh using its filename suffix."""

        suffix = Path(path).suffix.lower()
        if suffix == ".obj":
            return cls.from_obj(path)
        if suffix in {".stl", ".stla", ".stlb"}:
            return cls.from_stl(path)
        raise ValueError("mesh format must use an .obj or .stl extension")


def _obstacle_dims(obstacle):
    if isinstance(obstacle, MeshObstacle):
        return 3
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
    if any(isinstance(obstacle, MeshObstacle) for obstacle in obstacles):
        return None
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


def _point_triangle_distance(point, triangle):
    """Return the Euclidean distance from a point to a triangle."""

    a, b, c = np.asarray(triangle, dtype=float)
    point = np.asarray(point, dtype=float)
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(ap))
    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(bp))
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        value = d1 / (d1 - d3)
        return float(np.linalg.norm(point - (a + value * ab)))
    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(cp))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        value = d2 / (d2 - d6)
        return float(np.linalg.norm(point - (a + value * ac)))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        value = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return float(np.linalg.norm(point - (b + value * (c - b))))
    normal = np.cross(ab, ac)
    norm = float(np.linalg.norm(normal))
    return float(abs(np.dot(point - a, normal)) / norm) if norm > 1e-12 else float(np.linalg.norm(ap))


def _segment_triangle_intersects(start, end, triangle):
    """Return whether a segment intersects a triangle."""

    a, b, c = np.asarray(triangle, dtype=float)
    direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    edge1 = b - a
    edge2 = c - a
    h = np.cross(direction, edge2)
    determinant = float(np.dot(edge1, h))
    if abs(determinant) <= 1e-12:
        return _point_triangle_distance(start, triangle) <= 1e-12 or _point_triangle_distance(end, triangle) <= 1e-12
    inverse = 1.0 / determinant
    s = np.asarray(start, dtype=float) - a
    u = inverse * float(np.dot(s, h))
    if u < -1e-12 or u > 1.0 + 1e-12:
        return False
    q = np.cross(s, edge1)
    v = inverse * float(np.dot(direction, q))
    if v < -1e-12 or u + v > 1.0 + 1e-12:
        return False
    t = inverse * float(np.dot(edge2, q))
    return -1e-12 <= t <= 1.0 + 1e-12


def _segment_triangle_distance(start, end, triangle):
    if _segment_triangle_intersects(start, end, triangle):
        return 0.0
    vertices = np.asarray(triangle, dtype=float)
    return float(min(
        _point_triangle_distance(start, triangle),
        _point_triangle_distance(end, triangle),
        point_segment_distance(vertices[0], start, end),
        point_segment_distance(vertices[1], start, end),
        point_segment_distance(vertices[2], start, end),
    ))


def _point_inside_mesh(point, mesh):
    """Use odd/even ray casting for closed triangular meshes."""

    point = np.asarray(point, dtype=float)
    if np.any(point < mesh.lower - 1e-12) or np.any(point > mesh.upper + 1e-12):
        return False
    direction = np.array([1.0, 0.1234567, 0.2345671], dtype=float)
    direction /= np.linalg.norm(direction)
    count = 0
    for triangle in mesh.triangles:
        if _segment_triangle_intersects(point, point + direction * 1e6, triangle):
            count += 1
    return bool(count % 2)


def mesh_obstacle_clearance(mesh, obstacle):
    """Return conservative clearance between a mesh and another obstacle."""

    if not isinstance(mesh, MeshObstacle):
        raise TypeError("mesh must be a MeshObstacle")
    if isinstance(obstacle, MeshObstacle):
        lower = np.maximum(mesh.lower, obstacle.lower)
        upper = np.minimum(mesh.upper, obstacle.upper)
        if np.all(lower <= upper):
            return -0.0
        return float(min(
            min(point_obstacle_clearance(vertex, obstacle) for vertex in mesh.vertices),
            min(point_obstacle_clearance(vertex, mesh) for vertex in obstacle.vertices),
        ))
    if hasattr(obstacle, "center"):
        center = np.asarray(obstacle.center, dtype=float)
        if _point_inside_mesh(center, mesh):
            return -float(obstacle.radius)
        return float(
            min(_point_triangle_distance(center, triangle) for triangle in mesh.triangles)
            - float(obstacle.radius)
        )
    if hasattr(obstacle, "start") and hasattr(obstacle, "end"):
        midpoint = 0.5 * (np.asarray(obstacle.start) + np.asarray(obstacle.end))
        if _point_inside_mesh(midpoint, mesh):
            return -float(obstacle.radius)
        return float(
            min(_segment_triangle_distance(obstacle.start, obstacle.end, triangle) for triangle in mesh.triangles)
            - float(obstacle.radius)
        )
    if hasattr(obstacle, "lower") and hasattr(obstacle, "upper"):
        lower = np.asarray(obstacle.lower, dtype=float)
        upper = np.asarray(obstacle.upper, dtype=float)
        if np.all(mesh.lower <= upper) and np.all(mesh.upper >= lower):
            return -0.0
        corners = np.asarray(np.meshgrid(*zip(lower, upper), indexing="ij"), dtype=float).reshape(3, -1).T
        return float(min(
            min(point_obstacle_clearance(vertex, obstacle) for vertex in mesh.vertices),
            min(point_obstacle_clearance(corner, mesh) for corner in corners),
        ))
    raise TypeError("obstacle must be a supported obstacle type")


def mesh_collision_report(meshes, obstacles, *, link_radius=0.0):
    """Report clearance between transformed link meshes and world obstacles."""

    meshes = list(meshes)
    obstacles = list(obstacles)
    link_radius = float(link_radius)
    if link_radius < 0.0 or not np.isfinite(link_radius):
        raise ValueError("link_radius must be a non-negative finite value")
    closest: dict[str, Any] = {
        "kind": None,
        "index": None,
        "obstacle_index": None,
        "clearance": float("inf"),
    }
    hits = []
    for mesh_index, mesh in enumerate(meshes):
        for obstacle_index, obstacle in enumerate(obstacles):
            clearance = mesh_obstacle_clearance(mesh, obstacle) - link_radius
            if clearance < closest["clearance"]:
                closest = {
                    "kind": "mesh",
                    "index": mesh_index,
                    "obstacle_index": obstacle_index,
                    "clearance": float(clearance),
                }
            if clearance <= 0.0:
                hits.append({
                    "kind": "mesh",
                    "mesh_index": mesh_index,
                    "obstacle_index": obstacle_index,
                    "clearance": float(clearance),
                    "obstacle": obstacle,
                })
    return RobotResult({
        "collision": bool(hits),
        "minimum_clearance": float(closest["clearance"]),
        "closest": closest,
        "hits": hits,
    })


def point_obstacle_clearance(point, obstacle):
    """Return signed point clearance to an obstacle.

    Positive values are separated, zero touches, and negative values overlap.
    """

    dims = _obstacle_dims(obstacle)
    point = _point(point, dims, "point")
    if isinstance(obstacle, MeshObstacle):
        distances = np.asarray(
            [_point_triangle_distance(point, triangle) for triangle in obstacle.triangles],
            dtype=float,
        )
        distance = float(np.min(distances))
        return -distance if _point_inside_mesh(point, obstacle) else distance
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
    if isinstance(obstacle, MeshObstacle):
        clearances = []
        for triangle in obstacle.triangles:
            if _segment_triangle_intersects(start, end, triangle):
                return -0.0
            clearances.append(_segment_triangle_distance(start, end, triangle))
        midpoint = 0.5 * (start + end)
        distance = float(np.min(clearances))
        return -distance if _point_inside_mesh(midpoint, obstacle) else distance
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


def chain_self_collision_report(points, *, link_radius=0.0, exclusions=()):
    """Report clearance between non-adjacent chain link segments.

    Adjacent links are excluded by definition. ``exclusions`` may contain
    additional ``(segment_index, segment_index)`` pairs for robot-specific
    geometry that is known to overlap without being a collision.
    """

    points = _as_points(points)
    link_radius = float(link_radius)
    if link_radius < 0.0 or not np.isfinite(link_radius):
        raise ValueError("link_radius must be a non-negative finite value")
    n_segments = max(0, points.shape[0] - 1)
    excluded = set()
    for pair in exclusions:
        values = tuple(int(value) for value in pair)
        if len(values) != 2 or min(values) < 0 or max(values) >= n_segments:
            raise ValueError("self-collision exclusions must contain valid segment-index pairs")
        excluded.add(tuple(sorted(values)))

    closest: dict[str, object] = {
        "kind": "self",
        "index": None,
        "obstacle_index": None,
        "link_pair": None,
        "clearance": float("inf"),
    }
    closest_clearance = float("inf")
    hits = []
    collision_tolerance = 1e-12
    for first in range(n_segments):
        for second in range(first + 2, n_segments):
            if (first, second) in excluded:
                continue
            clearance = segment_distance(
                points[first], points[first + 1], points[second], points[second + 1]
            ) - 2.0 * link_radius
            if clearance < closest_clearance:
                closest = {
                    "kind": "self",
                    "index": first,
                    "obstacle_index": None,
                    "link_pair": (first, second),
                    "clearance": float(clearance),
                }
                closest_clearance = float(clearance)
            if clearance <= collision_tolerance:
                hits.append({
                    "kind": "self",
                    "link_pair": (first, second),
                    "clearance": float(clearance),
                })
    return RobotResult({
        "collision": bool(hits),
        "self_collision": bool(hits),
        "minimum_clearance": closest_clearance,
        "closest": closest,
        "hits": hits,
    })


def robot_collision_report(
    robot,
    joint_values=None,
    obstacles: Iterable[object] = (),
    *,
    link_radius=0.0,
    occupancy_grid=None,
    check_self_collision=False,
    self_collision_exclusions=(),
):
    """Return collision diagnostics for a robot model at one configuration."""

    frame_positions = getattr(robot, "frame_positions", robot.positions)
    chain = frame_positions(joint_values)
    report = chain_collision_report(chain, obstacles, link_radius=link_radius)
    self_report = chain_self_collision_report(
        chain,
        link_radius=link_radius,
        exclusions=self_collision_exclusions,
    ) if check_self_collision else None
    report["self_collision"] = False if self_report is None else self_report["self_collision"]
    report["self_collision_minimum_clearance"] = (
        float("inf") if self_report is None else self_report["minimum_clearance"]
    )
    if self_report is not None and self_report["minimum_clearance"] < report["minimum_clearance"]:
        report["minimum_clearance"] = self_report["minimum_clearance"]
        report["closest"] = self_report["closest"]
    if self_report is not None:
        report["hits"].extend(self_report["hits"])
        report["collision"] = bool(report["collision"] or self_report["collision"])
    if occupancy_grid is None:
        return report
    occupancy_clearance = occupancy_grid.polyline_minimum_clearance(chain, radius=link_radius)
    if occupancy_clearance < report["minimum_clearance"]:
        report["minimum_clearance"] = float(occupancy_clearance)
        report["closest"] = {
            "kind": "occupancy",
            "index": None,
            "obstacle_index": None,
            "clearance": float(occupancy_clearance),
        }
    report["occupancy_clearance"] = float(occupancy_clearance)
    report["collision"] = bool(report["collision"] or occupancy_clearance <= 0.0)
    return report


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


def _path_samples(robot, joint_path, interpolation_step):
    path = np.asarray(joint_path, dtype=float)
    if path.ndim != 2 or path.shape[0] < 1 or path.shape[1] != robot.n_joints:
        raise ValueError("joint_path must have shape (n_points, n_joints)")
    if not np.all(np.isfinite(path)):
        raise ValueError("joint_path must contain only finite values")
    interpolation_step = float(interpolation_step)
    if interpolation_step <= 0.0 or not np.isfinite(interpolation_step):
        raise ValueError("interpolation_step must be a positive finite value")
    return _interpolate_joint_path(path, interpolation_step)


def path_collision_free(
    robot,
    joint_path,
    obstacles: Iterable[object],
    *,
    clearance=0.0,
    link_radius=0.0,
    interpolation_step=0.05,
    occupancy_grid=None,
    check_self_collision=False,
    self_collision_exclusions=(),
):
    """Return whether every interpolated path sample meets ``clearance``.

    Chain frames are generated in one batch, then the compiled AABB
    broad-phase rejects distant obstacles before evaluating exact geometry.
    """

    samples = _path_samples(robot, joint_path, interpolation_step)
    points = robot.frame_positions_batch(samples)
    geometric_free = chain_collision_free_batch(
        points,
        obstacles,
        clearance=clearance,
        link_radius=link_radius,
    )
    if occupancy_grid is None:
        occupancy_free = np.ones(samples.shape[0], dtype=bool)
    else:
        occupancy_free = np.asarray(
            [
                occupancy_grid.polyline_collision_free(chain, radius=link_radius)
                for chain in points
            ],
            dtype=bool,
        )
    if check_self_collision:
        self_free = np.asarray([
            not chain_self_collision_report(
                chain,
                link_radius=link_radius,
                exclusions=self_collision_exclusions,
            )["collision"]
            for chain in points
        ], dtype=bool)
    else:
        self_free = np.ones(samples.shape[0], dtype=bool)
    return bool(np.all(geometric_free & occupancy_free & self_free))


def path_collision_summary(
    robot,
    joint_path,
    obstacles: Iterable[object],
    *,
    link_radius=0.0,
    interpolation_step=0.05,
    occupancy_grid=None,
    check_self_collision=False,
    self_collision_exclusions=(),
):
    """Return batched collision diagnostics along an interpolated joint path.

    This lightweight alternative to :func:`path_collision_report` keeps the
    per-sample results as typed arrays and routes kinematics and collision
    summaries through their optional compiled batch kernels.
    """

    samples = _path_samples(robot, joint_path, interpolation_step)
    points = robot.frame_positions_batch(samples)
    summary = chain_collision_summary_batch(points, obstacles, link_radius=link_radius)
    clearances = summary["minimum_clearance"]
    if occupancy_grid is not None:
        occupancy_clearances = np.asarray(
            [
                occupancy_grid.minimum_clearance(chain, radius=link_radius)
                for chain in points
            ],
            dtype=float,
        )
        clearances = np.minimum(clearances, occupancy_clearances)
        summary["collision"] = np.asarray(summary["collision"], dtype=bool) | (
            occupancy_clearances <= 0.0
        )
    else:
        occupancy_clearances = None
    self_clearances = None
    if check_self_collision:
        self_clearances = np.asarray([
            chain_self_collision_report(
                chain,
                link_radius=link_radius,
                exclusions=self_collision_exclusions,
            )["minimum_clearance"]
            for chain in points
        ], dtype=float)
        clearances = np.minimum(clearances, self_clearances)
        summary["collision"] = np.asarray(summary["collision"], dtype=bool) | (
            self_clearances <= 1e-12
        )
    collision_indices = np.flatnonzero(summary["collision"])
    finite = clearances[np.isfinite(clearances)]
    result = {
        "collision": bool(collision_indices.size),
        "minimum_clearance": float("inf") if finite.size == 0 else float(np.min(finite)),
        "first_collision_index": None if collision_indices.size == 0 else int(collision_indices[0]),
        "n_samples": samples.shape[0],
        "samples": samples,
        "clearances": clearances,
        "closest_kind": summary["closest_kind"],
        "closest_index": summary["closest_index"],
        "closest_obstacle_index": summary["closest_obstacle_index"],
    }
    if occupancy_clearances is not None:
        result["occupancy_clearances"] = occupancy_clearances
    if self_clearances is not None:
        result["self_collision_clearances"] = self_clearances
        result["self_collision"] = bool(np.any(self_clearances <= 1e-12))
    else:
        result["self_collision"] = False
    return RobotResult(result)


def path_collision_report(
    robot,
    joint_path,
    obstacles: Iterable[object],
    *,
    link_radius=0.0,
    interpolation_step=0.05,
    occupancy_grid=None,
    check_self_collision=False,
    self_collision_exclusions=(),
):
    """Return collision and clearance diagnostics along a joint-space path."""

    obstacles = list(obstacles)
    samples = _path_samples(robot, joint_path, interpolation_step)
    reports = []
    for q in samples:
        reports.append(
            robot_collision_report(
                robot,
                q,
                obstacles,
                link_radius=link_radius,
                occupancy_grid=occupancy_grid,
                check_self_collision=check_self_collision,
                self_collision_exclusions=self_collision_exclusions,
            )
        )
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
    "MeshObstacle",
    "chain_clearance_batch",
    "chain_collision_free_batch",
    "chain_collision_details_batch",
    "chain_collision_summary_batch",
    "SphereObstacle",
    "chain_collision_report",
    "chain_self_collision_report",
    "path_collision_free",
    "path_collision_summary",
    "path_collision_report",
    "mesh_collision_report",
    "mesh_obstacle_clearance",
    "point_obstacle_clearance",
    "point_segment_distance",
    "robot_collision_report",
    "segment_distance",
    "segment_obstacle_clearance",
]
