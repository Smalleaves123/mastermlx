"""Geometry helpers for task-level inspection simulation."""

from __future__ import annotations

import numpy as np

from .collision import BoxObstacle, CapsuleObstacle, MeshObstacle, SphereObstacle, segment_obstacle_clearance


def normalize_scan_targets(targets):
    """Normalize ordered scan targets to finite 3-vectors or 4x4 poses."""

    targets = list(targets)
    if not targets:
        raise ValueError("scan_poses must be non-empty")
    normalized = []
    shape = None
    for target in targets:
        value = np.asarray(target, dtype=float)
        if value.shape == (2,):
            value = np.concatenate([value, [0.0]])
        if value.shape not in {(3,), (4, 4)} or not np.all(np.isfinite(value)):
            raise ValueError("each scan pose must be a finite 2D/3D position or 4x4 pose")
        if shape is None:
            shape = value.shape
        if value.shape != shape:
            raise ValueError("scan poses must all use the same position or pose format")
        if value.shape == (4, 4):
            if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0]):
                raise ValueError("scan pose bottom row must be [0, 0, 0, 1]")
            rotation = value[:3, :3]
            if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
                raise ValueError("scan pose rotation must be orthonormal")
        normalized.append(value.copy())
    return normalized


def normalize_inspection_points(points):
    """Normalize surface points to an ``(n_points, 3)`` array."""

    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] not in {2, 3}:
        raise ValueError("inspection_points must have shape (n_points, 2 or 3)")
    if not np.all(np.isfinite(values)):
        raise ValueError("inspection_points must contain only finite values")
    if values.shape[1] == 2:
        values = np.column_stack([values, np.zeros(values.shape[0])])
    return values


def _target_position(target):
    return target if target.shape == (3,) else target[:3, 3]


def _obstacle_in_3d(obstacle):
    """Lift legacy planar obstacles into the z=0 inspection plane."""

    if isinstance(obstacle, MeshObstacle):
        return obstacle
    if hasattr(obstacle, "center"):
        center = tuple(obstacle.center)
        return obstacle if len(center) == 3 else SphereObstacle(center + (0.0,), obstacle.radius)
    if hasattr(obstacle, "lower") and hasattr(obstacle, "upper"):
        lower = tuple(obstacle.lower)
        upper = tuple(obstacle.upper)
        return obstacle if len(lower) == 3 else BoxObstacle(lower + (-1e-9,), upper + (1e-9,))
    if hasattr(obstacle, "start") and hasattr(obstacle, "end"):
        start = tuple(obstacle.start)
        end = tuple(obstacle.end)
        return obstacle if len(start) == 3 else CapsuleObstacle(start + (0.0,), end + (0.0,), obstacle.radius)
    raise TypeError("obstacle must be a supported geometry")


def _line_of_sight(start, end, obstacles):
    for obstacle in obstacles:
        if segment_obstacle_clearance(start, end, _obstacle_in_3d(obstacle)) <= 0.0:
            return False
    return True


def evaluate_inspection_coverage(
    scan_poses,
    inspection_points,
    reachable,
    obstacles=(),
    *,
    coverage_radius,
    min_range=0.0,
    max_range=np.inf,
    field_of_view=None,
    optical_axis=(0.0, 0.0, 1.0),
):
    """Evaluate point coverage, camera visibility, and obstacle occlusion."""

    points = normalize_inspection_points(inspection_points)
    reachable = np.asarray(reachable, dtype=bool).reshape(-1)
    if reachable.shape != (len(scan_poses),):
        raise ValueError("reachable must contain one flag per scan pose")
    coverage_radius = float(coverage_radius)
    min_range = float(min_range)
    max_range = float(max_range)
    if coverage_radius <= 0.0 or not np.isfinite(coverage_radius):
        raise ValueError("coverage_radius must be positive and finite")
    if min_range < 0.0 or not np.isfinite(min_range):
        raise ValueError("min_range must be non-negative and finite")
    if max_range <= 0.0 or np.isnan(max_range) or min_range > max_range:
        raise ValueError("max_range must be positive and at least min_range")
    axis = np.asarray(optical_axis, dtype=float).reshape(-1)
    if axis.size != 3 or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) == 0.0:
        raise ValueError("optical_axis must be a non-zero finite 3-vector")
    axis = axis / np.linalg.norm(axis)
    if field_of_view is not None:
        field_of_view = float(field_of_view)
        if field_of_view <= 0.0 or field_of_view > 180.0 or not np.isfinite(field_of_view):
            raise ValueError("field_of_view must be in (0, 180] degrees")
        cosine_limit = np.cos(np.deg2rad(field_of_view) / 2.0)
    else:
        cosine_limit = None

    covered = np.zeros(points.shape[0], dtype=bool)
    occluded = np.zeros(points.shape[0], dtype=bool)
    covered_by = np.full(points.shape[0], -1, dtype=int)
    for point_index, point in enumerate(points):
        candidate = False
        blocked = False
        for scan_index, (target, is_reachable) in enumerate(zip(scan_poses, reachable)):
            if not is_reachable:
                continue
            sensor_position = _target_position(target)
            vector = point - sensor_position
            distance = float(np.linalg.norm(vector))
            if distance < min_range or distance > max_range or distance > coverage_radius:
                continue
            if cosine_limit is not None and target.shape == (4, 4) and distance > 1e-12:
                direction = target[:3, :3] @ axis
                if float(np.dot(direction, vector) / distance) < cosine_limit:
                    continue
            candidate = True
            if _line_of_sight(sensor_position, point, obstacles):
                covered[point_index] = True
                covered_by[point_index] = scan_index
                break
            blocked = True
        occluded[point_index] = bool(candidate and blocked and not covered[point_index])
    return {
        "covered": covered,
        "occluded": occluded,
        "covered_by_scan_index": covered_by,
        "coverage_rate": float(np.mean(covered)),
        "occlusion_rate": float(np.mean(occluded)),
    }


__all__ = ["evaluate_inspection_coverage", "normalize_inspection_points", "normalize_scan_targets"]
