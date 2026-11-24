"""Persistent single-scale local occupancy built from the 96 range rays."""

from __future__ import annotations

import numpy as np


UNKNOWN = np.int8(0)
FREE = np.int8(-1)
OCCUPIED = np.int8(1)


class PersistentLocalOccupancyMap:
    """Sparse world-frame map with a UAV-centred fixed-size output window.

    The sparse cells persist while they remain inside the sliding window.
    Ray interiors become FREE, finite hit endpoints become OCCUPIED, and cells
    never traversed by a ray remain UNKNOWN.  The flattened representation is
    a single tri-state channel: -1 free, 0 unknown, +1 occupied.
    """

    def __init__(self, *, voxel_size: float = 0.4, shape=(12, 12, 6)):
        self.voxel_size = float(voxel_size)
        self.shape = tuple(int(value) for value in shape)
        if self.voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive.")
        if len(self.shape) != 3 or any(value <= 0 for value in self.shape):
            raise ValueError("shape must contain three positive dimensions.")
        self._voxels: dict[tuple[int, int, int], np.int8] = {}
        self._last_center_index = np.zeros(3, dtype=np.int64)

    @property
    def observation_dim(self) -> int:
        return int(np.prod(self.shape))

    def reset(self) -> None:
        self._voxels.clear()
        self._last_center_index.fill(0)

    def world_to_index(self, point) -> np.ndarray:
        return np.floor(
            np.asarray(point, dtype=np.float64).reshape(3) / self.voxel_size
        ).astype(np.int64)

    def _window_bounds(self, center_position) -> tuple[np.ndarray, np.ndarray]:
        center_index = self.world_to_index(center_position)
        lower = center_index - np.asarray(self.shape, dtype=np.int64) // 2
        upper = lower + np.asarray(self.shape, dtype=np.int64)
        return lower, upper

    def _inside(self, index, lower, upper) -> bool:
        value = np.asarray(index, dtype=np.int64)
        return bool(np.all(value >= lower) and np.all(value < upper))

    def _prune(self, center_position) -> None:
        lower, upper = self._window_bounds(center_position)
        self._last_center_index = self.world_to_index(center_position)
        self._voxels = {
            key: value
            for key, value in self._voxels.items()
            if self._inside(key, lower, upper)
        }

    def _ray_indices(self, start, end) -> list[tuple[int, int, int]]:
        start = np.asarray(start, dtype=np.float64).reshape(3)
        end = np.asarray(end, dtype=np.float64).reshape(3)
        distance = float(np.linalg.norm(end - start))
        samples = max(1, int(np.ceil(distance / (self.voxel_size * 0.35))))
        points = np.linspace(start, end, samples + 1, dtype=np.float64)
        indices = []
        previous = None
        for point in points:
            index = tuple(int(value) for value in self.world_to_index(point))
            if index != previous:
                indices.append(index)
                previous = index
        return indices

    def update_from_rays(
        self,
        *,
        position,
        directions,
        range_fractions,
        sensor_range: float,
        origin_offset: float,
        vertical_origin_offset: float = 0.02,
    ) -> None:
        position = np.asarray(position, dtype=np.float64).reshape(3)
        directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        fractions = np.asarray(range_fractions, dtype=np.float64).reshape(-1)
        if len(directions) != len(fractions):
            raise ValueError("directions and range_fractions must have equal length.")
        if sensor_range <= 0.0 or not 0.0 <= origin_offset < sensor_range:
            raise ValueError("Ray range and origin offset are inconsistent.")

        self._prune(position)
        lower, upper = self._window_bounds(position)
        centre = position + np.array([0.0, 0.0, vertical_origin_offset])
        occupied_updates = set()
        free_updates = set()
        for direction, fraction in zip(directions, fractions):
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-8:
                continue
            direction = direction / norm
            fraction = float(np.clip(fraction, 0.0, 1.0))
            distance = fraction * float(sensor_range)
            hit = fraction < 1.0 - 1e-6
            start = centre + direction * float(origin_offset)
            endpoint = centre + direction * distance
            indices = self._ray_indices(start, endpoint)
            if hit and indices:
                occupied_index = indices[-1]
                if self._inside(occupied_index, lower, upper):
                    occupied_updates.add(occupied_index)
                indices = indices[:-1]
            for index in indices:
                if self._inside(index, lower, upper):
                    free_updates.add(index)

        # Occupied wins if coarse discretisation lets another ray traverse the
        # same voxel during this or a later update.
        for index in free_updates:
            if self._voxels.get(index) != OCCUPIED and index not in occupied_updates:
                self._voxels[index] = FREE
        for index in occupied_updates:
            self._voxels[index] = OCCUPIED

    def local_grid(self, center_position) -> np.ndarray:
        self._prune(center_position)
        lower, upper = self._window_bounds(center_position)
        grid = np.full(self.shape, UNKNOWN, dtype=np.int8)
        for key, value in self._voxels.items():
            index = np.asarray(key, dtype=np.int64)
            if self._inside(index, lower, upper):
                local = tuple((index - lower).tolist())
                grid[local] = value
        return grid

    def observation(self, center_position) -> np.ndarray:
        return self.local_grid(center_position).astype(np.float32).reshape(-1)

    def statistics(self, center_position) -> dict:
        grid = self.local_grid(center_position)
        return {
            "unknown": int(np.count_nonzero(grid == UNKNOWN)),
            "free": int(np.count_nonzero(grid == FREE)),
            "occupied": int(np.count_nonzero(grid == OCCUPIED)),
        }
