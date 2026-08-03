"""Small NumPy-only voxel occupancy maps for robotics workflows."""

from __future__ import annotations

import numpy as np


def _points(values, *, allow_empty=False):
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        if array.size != 3:
            raise ValueError("points must contain 3D coordinates")
        array = array.reshape(1, 3)
    if array.ndim != 2 or array.shape[1] != 3 or (not allow_empty and array.shape[0] < 1):
        raise ValueError("points must have shape (n_points, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError("points must contain only finite values")
    return array


class VoxelOccupancyGrid:
    """Axis-aligned 3D occupancy grid.

    ``bounds`` is ``(lower, upper)`` and the upper edge is exclusive.  Grid
    cells are addressed in ``(x, y, z)`` order, and ``grid_to_world`` returns
    cell centers.  Queries outside the bounds are treated as free by
    ``is_occupied`` and collision methods, while ``world_to_grid`` raises a
    clear error so accidental map writes cannot silently wrap around.
    """

    def __init__(self, bounds, resolution, *, occupied=None):
        bounds_array = np.asarray(bounds, dtype=float)
        if bounds_array.shape != (2, 3) or not np.all(np.isfinite(bounds_array)):
            raise ValueError("bounds must have shape (2, 3) and contain finite values")
        lower, upper = bounds_array
        if np.any(upper <= lower):
            raise ValueError("bounds upper values must be strictly above lower values")
        resolution_array = np.asarray(resolution, dtype=float)
        if resolution_array.ndim == 0:
            resolution_array = np.full(3, float(resolution_array))
        if resolution_array.shape != (3,) or not np.all(np.isfinite(resolution_array)):
            raise ValueError("resolution must be a positive scalar or a 3-vector")
        if np.any(resolution_array <= 0.0):
            raise ValueError("resolution must be strictly positive")
        shape = np.ceil((upper - lower) / resolution_array).astype(np.int64)
        if np.any(shape < 1):
            raise ValueError("bounds must contain at least one voxel per dimension")
        self.bounds = bounds_array.copy()
        self.resolution = (
            float(resolution_array[0])
            if np.allclose(resolution_array, resolution_array[0])
            else resolution_array.copy()
        )
        self.shape = tuple(int(value) for value in shape)
        if occupied is None:
            self.occupied = np.zeros(self.shape, dtype=bool)
        else:
            occupied_array = np.asarray(occupied, dtype=bool)
            if occupied_array.shape != self.shape:
                raise ValueError(f"occupied must have shape {self.shape}")
            self.occupied = occupied_array.copy()

    @property
    def lower(self):
        """Return the lower world bound."""

        return self.bounds[0]

    @property
    def upper(self):
        """Return the upper world bound."""

        return self.bounds[1]

    @property
    def voxel_diagonal(self):
        """Return the length of a voxel diagonal."""

        return float(np.linalg.norm(self._resolution_array))

    @property
    def _resolution_array(self):
        value = np.asarray(self.resolution, dtype=float)
        return np.full(3, float(value)) if value.ndim == 0 else value

    def _world_indices(self, points):
        values = _points(points)
        resolution = self._resolution_array
        indices = np.floor((values - self.lower) / resolution).astype(np.int64)
        valid = np.all((values >= self.lower) & (values < self.upper), axis=1)
        return values, indices, valid

    def world_to_grid(self, points):
        """Convert one or more in-bounds world points to integer voxel indices."""

        values, indices, valid = self._world_indices(points)
        if not np.all(valid):
            raise ValueError("points must lie inside the occupancy grid bounds")
        return indices[0] if values.shape[0] == 1 else indices

    def grid_to_world(self, indices):
        """Convert one or more voxel indices to world-space cell centers."""

        values = np.asarray(indices, dtype=float)
        single = values.ndim == 1
        if single:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
            raise ValueError("indices must have shape (n_indices, 3)")
        integer = values.astype(np.int64)
        if not np.all(values == integer) or np.any(integer < 0) or np.any(integer >= np.asarray(self.shape)):
            raise ValueError("indices must refer to voxels inside the occupancy grid")
        centers = self.lower + (integer + 0.5) * self._resolution_array
        return centers[0] if single else centers

    def mark_occupied(self, points):
        """Mark all in-bounds point-containing voxels as occupied."""

        _, indices, valid = self._world_indices(points)
        for index in indices[valid]:
            self.occupied[tuple(index)] = True
        return self

    def clear_occupied(self, points):
        """Clear all in-bounds point-containing voxels."""

        _, indices, valid = self._world_indices(points)
        for index in indices[valid]:
            self.occupied[tuple(index)] = False
        return self

    def is_occupied(self, points):
        """Query occupancy for one or more world points."""

        values, indices, valid = self._world_indices(points)
        result = np.zeros(values.shape[0], dtype=bool)
        for position, (index, is_valid) in enumerate(zip(indices, valid)):
            if is_valid:
                result[position] = self.occupied[tuple(index)]
        return bool(result[0]) if values.shape[0] == 1 else result

    def _nearby_occupied(self, point, radius):
        _, indices, valid = self._world_indices(point)
        if not valid[0]:
            return False
        radius = float(radius)
        margin = radius + 0.5 * self.voxel_diagonal
        minimum = np.floor((point - margin - self.lower) / self._resolution_array).astype(int)
        maximum = np.floor((point + margin - self.lower) / self._resolution_array).astype(int)
        minimum = np.maximum(minimum, 0)
        maximum = np.minimum(maximum, np.asarray(self.shape) - 1)
        if np.any(minimum > maximum):
            return False
        return bool(self.occupied[
            minimum[0] : maximum[0] + 1,
            minimum[1] : maximum[1] + 1,
            minimum[2] : maximum[2] + 1,
        ].any())

    def collision_free(self, points, radius=0.0):
        """Return whether all query points clear occupied voxels."""

        values = _points(points)
        radius = float(radius)
        if radius < 0.0 or not np.isfinite(radius):
            raise ValueError("radius must be a non-negative finite value")
        return not any(self._nearby_occupied(point, radius) for point in values)

    def minimum_clearance(self, points, radius=0.0):
        """Return a conservative signed clearance to the nearest occupied voxel."""

        values = _points(points)
        radius = float(radius)
        if radius < 0.0 or not np.isfinite(radius):
            raise ValueError("radius must be a non-negative finite value")
        occupied_indices = np.argwhere(self.occupied)
        if occupied_indices.size == 0:
            return float("inf")
        centers = self.grid_to_world(occupied_indices)
        clearances = []
        for point in values:
            if np.any(point < self.lower) or np.any(point >= self.upper):
                continue
            distance = np.min(np.linalg.norm(centers - point, axis=1))
            clearances.append(float(distance - 0.5 * self.voxel_diagonal - radius))
        return float("inf") if not clearances else float(np.min(clearances))

    def polyline_collision_free(self, points, radius=0.0):
        """Return whether a polyline clears occupied voxels.

        Segments are sampled at no more than half the smallest voxel edge,
        making this suitable for frame polylines returned by a robot model.
        """

        return self.collision_free(self._polyline_samples(points), radius=radius)

    def polyline_minimum_clearance(self, points, radius=0.0):
        """Return the conservative minimum clearance along a polyline."""

        return self.minimum_clearance(self._polyline_samples(points), radius=radius)

    def _polyline_samples(self, points):
        values = _points(points)
        if values.shape[0] == 1:
            return values
        spacing = 0.5 * float(np.min(self._resolution_array))
        samples = [values[0]]
        for start, end in zip(values[:-1], values[1:]):
            count = max(1, int(np.ceil(np.linalg.norm(end - start) / spacing)))
            samples.extend(
                start + alpha * (end - start)
                for alpha in np.linspace(0.0, 1.0, count + 1)[1:]
            )
        return np.asarray(samples)

    @classmethod
    def from_point_cloud(cls, points, *, bounds, resolution):
        """Voxelize a point cloud, ignoring samples outside ``bounds``."""

        values = _points(points, allow_empty=True)
        grid = cls(bounds, resolution)
        if values.shape[0]:
            grid.mark_occupied(values)
        return grid


__all__ = ["VoxelOccupancyGrid"]
