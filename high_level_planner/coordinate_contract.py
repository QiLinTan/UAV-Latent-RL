"""Planner-v2 heading-local coordinates; frozen Encoder input is unchanged."""

import numpy as np


def rotate_z(vectors, yaw):
    values = np.asarray(vectors, dtype=np.float64)
    if values.shape[-1] != 3 or not np.isfinite(values).all() or not np.isfinite(yaw):
        raise ValueError("Rotation requires finite xyz vectors and yaw")
    c, s = np.cos(yaw), np.sin(yaw)
    matrix = np.array([[c,-s,0], [s,c,0], [0,0,1]])
    return values @ matrix.T


def future_to_local(positions, velocities, origin, yaw):
    return np.concatenate([rotate_z(np.asarray(positions)-np.asarray(origin), -yaw),
                           rotate_z(velocities, -yaw)], axis=-1)


def future_to_world(sequence, origin, yaw):
    sequence = np.asarray(sequence)
    return rotate_z(sequence[..., :3], yaw)+np.asarray(origin), rotate_z(sequence[..., 3:], yaw)


def local_goal(goal, origin, yaw):
    # Use physical displacement, not the clipped normalized goal tail.
    return rotate_z(np.asarray(goal)-np.asarray(origin), -yaw)


def ranges_to_heading_bins(world_ranges, yaw):
    """Conservative adjacent-bin resampling, NOT new sensor measurements.

    Existing eight rays are world-fixed at j*pi/4. For arbitrary yaw, local
    bin j lies between two old rays: retain their minimum normalized range.
    This can miss an obstacle between rays; it is not a collision guarantee.
    Exact multiples of pi/4 are lossless permutations. No map access is used.
    """
    ranges = np.asarray(world_ranges, dtype=np.float64)
    if ranges.shape[-1] != 8 or not np.isfinite(ranges).all() or not np.isfinite(yaw):
        raise ValueError("Expected eight finite ranges and a finite yaw")
    if (ranges < 0).any() or (ranges > 1).any():
        raise ValueError("Ranges must be normalized to [0,1]")
    index = (np.arange(8) + yaw/(np.pi/4)) % 8
    lower = np.floor(index + 1e-12).astype(int) % 8
    upper = np.ceil(index - 1e-12).astype(int) % 8
    return np.minimum(ranges[...,lower], ranges[...,upper])
