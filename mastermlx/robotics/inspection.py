"""Camera geometry and greedy planning helpers for inspection simulation."""

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


def _unit_vector(values, name):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        raise ValueError(f"{name} must be non-zero")
    return values / norm


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
    return all(
        segment_obstacle_clearance(start, end, _obstacle_in_3d(obstacle)) > 0.0
        for obstacle in obstacles
    )


def look_at_pose(position, target, *, up=(0.0, 0.0, 1.0)):
    """Return a pose whose local +Z camera axis points at ``target``."""

    position = np.asarray(position, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)
    if position.shape != (3,) or target.shape != (3,):
        raise ValueError("position and target must be 3-vectors")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(target)):
        raise ValueError("position and target must contain only finite values")
    forward = _unit_vector(target - position, "target - position")
    up = _unit_vector(up, "up")
    if abs(float(np.dot(forward, up))) > 0.99:
        alternatives = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        up = next(axis for axis in alternatives if abs(float(np.dot(forward, axis))) <= 0.99)
    right = _unit_vector(np.cross(up, forward), "camera right")
    camera_up = np.cross(forward, right)
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = np.column_stack([right, camera_up, forward])
    pose[:3, 3] = position
    return pose


def generate_viewpoint_candidates(
    inspection_points,
    *,
    standoff,
    target=None,
    directions=None,
    up=(0.0, 0.0, 1.0),
):
    """Generate look-at camera poses around an inspection target.

    The default six axis directions provide a compact, deterministic candidate
    set. Callers can pass surface-normal or task-specific ``directions`` for a
    denser view lattice.
    """

    points = normalize_inspection_points(inspection_points)
    standoff = float(standoff)
    if standoff <= 0.0 or not np.isfinite(standoff):
        raise ValueError("standoff must be positive and finite")
    if target is None:
        target = np.mean(points, axis=0)
    else:
        target = np.asarray(target, dtype=float).reshape(-1)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target must be a finite 3-vector")
    if directions is None:
        directions = np.vstack([np.eye(3), -np.eye(3)])
    directions = np.asarray(directions, dtype=float)
    if directions.ndim != 2 or directions.shape[0] < 1 or directions.shape[1] != 3:
        raise ValueError("directions must have shape (n_directions, 3)")
    normalized = np.asarray([_unit_vector(direction, "direction") for direction in directions])
    return [look_at_pose(target + standoff * direction, target, up=up) for direction in normalized]


def _validate_ranges(coverage_radius, min_range, max_range):
    coverage_radius = float(coverage_radius)
    min_range = float(min_range)
    max_range = float(max_range)
    if coverage_radius <= 0.0 or not np.isfinite(coverage_radius):
        raise ValueError("coverage_radius must be positive and finite")
    if min_range < 0.0 or not np.isfinite(min_range):
        raise ValueError("min_range must be non-negative and finite")
    if max_range <= 0.0 or np.isnan(max_range) or min_range > max_range:
        raise ValueError("max_range must be positive and at least min_range")
    return coverage_radius, min_range, max_range


def _validate_fov(value, name):
    if value is None:
        return None
    value = float(value)
    if value <= 0.0 or value > 180.0 or not np.isfinite(value):
        raise ValueError(f"{name} must be in (0, 180] degrees")
    return value


def _frustum_contains(vector, pose, optical_axis, field_of_view, horizontal_fov, vertical_fov):
    if pose.shape != (4, 4):
        return True
    distance = float(np.linalg.norm(vector))
    if distance <= 1e-12:
        return True
    local = pose[:3, :3].T @ vector
    forward_axis = _unit_vector(optical_axis, "optical_axis")
    forward = float(np.dot(local, forward_axis))
    if forward <= 0.0:
        return False
    if field_of_view is not None:
        cosine_limit = np.cos(np.deg2rad(field_of_view) / 2.0)
        return forward / distance >= cosine_limit
    if horizontal_fov is None and vertical_fov is None:
        return True
    basis = np.eye(3)
    horizontal_axis = basis[np.argmin(np.abs(basis @ forward_axis))]
    horizontal_axis = _unit_vector(
        horizontal_axis - np.dot(horizontal_axis, forward_axis) * forward_axis,
        "horizontal camera axis",
    )
    vertical_axis = np.cross(forward_axis, horizontal_axis)
    horizontal = abs(float(np.dot(local, horizontal_axis)))
    vertical = abs(float(np.dot(local, vertical_axis)))
    if horizontal_fov is not None and horizontal > forward * np.tan(np.deg2rad(horizontal_fov) / 2.0):
        return False
    return vertical_fov is None or vertical <= forward * np.tan(np.deg2rad(vertical_fov) / 2.0)


def camera_visibility_matrix(
    scan_poses,
    inspection_points,
    reachable=None,
    obstacles=(),
    *,
    coverage_radius,
    min_range=0.0,
    max_range=np.inf,
    field_of_view=None,
    horizontal_field_of_view=None,
    vertical_field_of_view=None,
    optical_axis=(0.0, 0.0, 1.0),
):
    """Evaluate camera frustum, ray occlusion, and point visibility per view."""

    scan_poses = normalize_scan_targets(scan_poses)
    points = normalize_inspection_points(inspection_points)
    n_views = len(scan_poses)
    if reachable is None:
        reachable = np.ones(n_views, dtype=bool)
    reachable = np.asarray(reachable, dtype=bool).reshape(-1)
    if reachable.shape != (n_views,):
        raise ValueError("reachable must contain one flag per scan pose")
    coverage_radius, min_range, max_range = _validate_ranges(coverage_radius, min_range, max_range)
    field_of_view = _validate_fov(field_of_view, "field_of_view")
    horizontal_fov = _validate_fov(horizontal_field_of_view, "horizontal_field_of_view")
    vertical_fov = _validate_fov(vertical_field_of_view, "vertical_field_of_view")
    if field_of_view is not None and (horizontal_fov is not None or vertical_fov is not None):
        raise ValueError("field_of_view cannot be combined with horizontal/vertical field of view")
    optical_axis = _unit_vector(optical_axis, "optical_axis")
    obstacles = tuple(obstacles)

    shape = (n_views, points.shape[0])
    distances = np.full(shape, np.inf, dtype=float)
    in_range = np.zeros(shape, dtype=bool)
    in_frustum = np.zeros(shape, dtype=bool)
    line_of_sight = np.zeros(shape, dtype=bool)
    visible = np.zeros(shape, dtype=bool)
    occluded = np.zeros(shape, dtype=bool)
    for view_index, pose in enumerate(scan_poses):
        sensor_position = _target_position(pose)
        for point_index, point in enumerate(points):
            vector = point - sensor_position
            distance = float(np.linalg.norm(vector))
            distances[view_index, point_index] = distance
            in_range[view_index, point_index] = (
                min_range <= distance <= max_range and distance <= coverage_radius
            )
            if not in_range[view_index, point_index]:
                continue
            in_frustum[view_index, point_index] = _frustum_contains(
                vector,
                pose,
                optical_axis,
                field_of_view,
                horizontal_fov,
                vertical_fov,
            )
            if not in_frustum[view_index, point_index]:
                continue
            line_of_sight[view_index, point_index] = _line_of_sight(sensor_position, point, obstacles)
            visible[view_index, point_index] = (
                reachable[view_index] and line_of_sight[view_index, point_index]
            )
            occluded[view_index, point_index] = (
                reachable[view_index] and not line_of_sight[view_index, point_index]
            )
    return {
        "distances": distances,
        "in_range": in_range,
        "in_frustum": in_frustum,
        "line_of_sight": line_of_sight,
        "visible": visible,
        "occluded": occluded,
    }


def select_inspection_viewpoints(
    visibility,
    scan_poses,
    *,
    reachable=None,
    point_weights=None,
    required_coverage=1.0,
    start_position=None,
    travel_speed=1.0,
    dwell_time=0.0,
    time_weight=0.0,
    max_views=None,
):
    """Greedily choose and order views for coverage with a travel-time cost.

    This is a deterministic weighted set-cover approximation. Occluded and
    unreachable candidates have zero gain, so the selected route jointly
    reflects visibility, reachability, coverage, and estimated scan time.
    """

    visible = np.asarray(visibility, dtype=bool)
    if visible.ndim != 2 or visible.shape[0] < 1 or visible.shape[1] < 1:
        raise ValueError("visibility must have shape (n_views, n_points)")
    poses = normalize_scan_targets(scan_poses)
    if len(poses) != visible.shape[0]:
        raise ValueError("scan_poses must contain one pose per visibility row")
    n_views, n_points = visible.shape
    if reachable is None:
        reachable = np.ones(n_views, dtype=bool)
    reachable = np.asarray(reachable, dtype=bool).reshape(-1)
    if reachable.shape != (n_views,):
        raise ValueError("reachable must contain one flag per view")
    if point_weights is None:
        point_weights = np.ones(n_points, dtype=float)
    point_weights = np.asarray(point_weights, dtype=float).reshape(-1)
    if point_weights.shape != (n_points,) or not np.all(np.isfinite(point_weights)):
        raise ValueError("point_weights must contain one finite value per inspection point")
    if np.any(point_weights < 0.0) or not np.any(point_weights > 0.0):
        raise ValueError("point_weights must be non-negative with at least one positive value")
    required_coverage = float(required_coverage)
    if required_coverage < 0.0 or required_coverage > 1.0 or not np.isfinite(required_coverage):
        raise ValueError("required_coverage must be in [0, 1]")
    travel_speed = float(travel_speed)
    dwell_time = float(dwell_time)
    time_weight = float(time_weight)
    if travel_speed <= 0.0 or not np.isfinite(travel_speed):
        raise ValueError("travel_speed must be positive and finite")
    if dwell_time < 0.0 or not np.isfinite(dwell_time):
        raise ValueError("dwell_time must be non-negative and finite")
    if time_weight < 0.0 or not np.isfinite(time_weight):
        raise ValueError("time_weight must be non-negative and finite")
    if max_views is not None:
        max_views = int(max_views)
        if max_views < 1:
            raise ValueError("max_views must be at least 1")
    if start_position is not None:
        start_position = np.asarray(start_position, dtype=float).reshape(-1)
        if start_position.shape != (3,) or not np.all(np.isfinite(start_position)):
            raise ValueError("start_position must be a finite 3-vector")

    positions = np.asarray([_target_position(pose) for pose in poses], dtype=float)
    covered = np.zeros(n_points, dtype=bool)
    selected = []
    marginal_gain = []
    segment_distances = []
    current = start_position
    total_weight = float(np.sum(point_weights))
    while float(np.sum(point_weights[covered]) / total_weight) < required_coverage:
        if max_views is not None and len(selected) >= max_views:
            break
        best_index = None
        best_score = -np.inf
        best_gain = 0.0
        best_distance = 0.0
        for index in range(n_views):
            if index in selected or not reachable[index]:
                continue
            gain = float(np.sum(point_weights[visible[index] & ~covered]))
            if gain <= 0.0:
                continue
            distance = 0.0 if current is None else float(np.linalg.norm(positions[index] - current))
            visit_time = dwell_time + distance / travel_speed
            score = gain / (1.0 + time_weight * visit_time)
            if score > best_score + 1e-12 or (
                abs(score - best_score) <= 1e-12 and best_index is not None and index < best_index
            ):
                best_index = index
                best_score = score
                best_gain = gain
                best_distance = distance
        if best_index is None:
            break
        selected.append(best_index)
        marginal_gain.append(best_gain)
        segment_distances.append(best_distance)
        covered |= visible[best_index]
        current = positions[best_index]

    path_length = float(np.sum(segment_distances))
    total_time = path_length / travel_speed + dwell_time * len(selected)
    return {
        "selected_indices": np.asarray(selected, dtype=int),
        "covered": covered,
        "uncovered": ~covered,
        "coverage_rate": float(np.sum(point_weights[covered]) / total_weight),
        "marginal_gain": np.asarray(marginal_gain, dtype=float),
        "segment_distances": np.asarray(segment_distances, dtype=float),
        "path_length": path_length,
        "estimated_motion_time": path_length / travel_speed,
        "estimated_total_time": total_time,
        "required_coverage_met": float(np.sum(point_weights[covered]) / total_weight) >= required_coverage,
    }


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
    horizontal_field_of_view=None,
    vertical_field_of_view=None,
    optical_axis=(0.0, 0.0, 1.0),
):
    """Evaluate point coverage, camera visibility, and obstacle occlusion."""

    visibility = camera_visibility_matrix(
        scan_poses,
        inspection_points,
        reachable,
        obstacles,
        coverage_radius=coverage_radius,
        min_range=min_range,
        max_range=max_range,
        field_of_view=field_of_view,
        horizontal_field_of_view=horizontal_field_of_view,
        vertical_field_of_view=vertical_field_of_view,
        optical_axis=optical_axis,
    )
    covered = np.any(visibility["visible"], axis=0)
    occluded = np.any(visibility["occluded"], axis=0) & ~covered
    covered_by = np.full(covered.shape[0], -1, dtype=int)
    for point_index in np.flatnonzero(covered):
        covered_by[point_index] = int(np.flatnonzero(visibility["visible"][:, point_index])[0])
    return {
        "covered": covered,
        "occluded": occluded,
        "covered_by_scan_index": covered_by,
        "coverage_rate": float(np.mean(covered)),
        "occlusion_rate": float(np.mean(occluded)),
    }


__all__ = [
    "camera_visibility_matrix",
    "evaluate_inspection_coverage",
    "generate_viewpoint_candidates",
    "look_at_pose",
    "normalize_inspection_points",
    "normalize_scan_targets",
    "select_inspection_viewpoints",
]
