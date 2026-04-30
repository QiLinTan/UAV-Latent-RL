from __future__ import annotations

import numpy as np


def route_direction_xy(start_xy, goal_xy):
    start_xy = np.asarray(start_xy, dtype=np.float32)
    goal_xy = np.asarray(goal_xy, dtype=np.float32)
    route = goal_xy - start_xy
    norm = float(np.linalg.norm(route))
    if norm < 1e-8:
        return np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)
    route_dir = route / norm
    perp_dir = np.array([-route_dir[1], route_dir[0]], dtype=np.float32)
    return route_dir.astype(np.float32), perp_dir


def distance_point_to_segment_2d(point, seg_a, seg_b):
    point = np.asarray(point, dtype=np.float32)
    seg_a = np.asarray(seg_a, dtype=np.float32)
    seg_b = np.asarray(seg_b, dtype=np.float32)
    ab = seg_b - seg_a
    denom = float(np.dot(ab, ab))
    if denom < 1e-8:
        return float(np.linalg.norm(point - seg_a))
    t = float(np.dot(point - seg_a, ab) / denom)
    t = np.clip(t, 0.0, 1.0)
    proj = seg_a + t * ab
    return float(np.linalg.norm(point - proj))


def distance_point_to_line_2d(point, line_a, line_b):
    point = np.asarray(point, dtype=np.float32)
    line_a = np.asarray(line_a, dtype=np.float32)
    line_b = np.asarray(line_b, dtype=np.float32)
    ab = line_b - line_a
    denom = float(np.dot(ab, ab))
    if denom < 1e-8:
        return float(np.linalg.norm(point - line_a))
    t = float(np.dot(point - line_a, ab) / denom)
    proj = line_a + t * ab
    return float(np.linalg.norm(point - proj))


def route_projection_2d(point, line_a, line_b):
    point = np.asarray(point, dtype=np.float32)
    line_a = np.asarray(line_a, dtype=np.float32)
    line_b = np.asarray(line_b, dtype=np.float32)
    ab = line_b - line_a
    denom = float(np.linalg.norm(ab))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(point - line_a, ab / denom))
