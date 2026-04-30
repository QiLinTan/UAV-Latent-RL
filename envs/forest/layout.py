from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p

from .geometry import distance_point_to_segment_2d, route_direction_xy


@dataclass(frozen=True)
class ForestLayoutConfig:
    forest_half_extent: float
    num_trees: int
    tree_radius_range: tuple[float, float]
    tree_height_range: tuple[float, float]
    min_tree_separation: float
    centerline_band_width: float


class ForestLayoutGenerator:
    def __init__(self, config: ForestLayoutConfig):
        self.config = config

    def generate(
        self,
        *,
        rng,
        start_pos,
        goal_pos,
        corridor_half_width: float,
        protect_corridor: bool,
        corridor_edge_tree_fraction: float,
        centerline_tree_fraction: float,
    ):
        start_pos = np.asarray(start_pos, dtype=np.float32)
        goal_pos = np.asarray(goal_pos, dtype=np.float32)
        tree_specs = []
        attempts = 0
        max_attempts = max(200, self.config.num_trees * 40)

        while len(tree_specs) < self.config.num_trees and attempts < max_attempts:
            attempts += 1
            radius = float(rng.uniform(*self.config.tree_radius_range))
            height = float(rng.uniform(*self.config.tree_height_range))
            xy = self._sample_tree_xy(
                rng=rng,
                radius=radius,
                start_pos=start_pos,
                goal_pos=goal_pos,
                corridor_half_width=corridor_half_width,
                protect_corridor=protect_corridor,
                corridor_edge_tree_fraction=corridor_edge_tree_fraction,
                centerline_tree_fraction=centerline_tree_fraction,
            )

            if not self._is_tree_placement_valid(
                xy=xy,
                radius=radius,
                start_pos=start_pos,
                goal_pos=goal_pos,
                tree_specs=tree_specs,
                corridor_half_width=corridor_half_width,
                protect_corridor=protect_corridor,
            ):
                continue

            tree_specs.append(
                {
                    "xy": xy,
                    "radius": radius,
                    "height": height,
                }
            )

        return tree_specs

    def create_tree_body(self, *, client_id: int, xy, radius: float, height: float):
        collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=radius,
            height=height,
            physicsClientId=client_id,
        )
        visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=[0.42, 0.26, 0.12, 1.0],
            physicsClientId=client_id,
        )
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[float(xy[0]), float(xy[1]), height / 2.0],
            physicsClientId=client_id,
        )

    @staticmethod
    def compute_nearest_tree_clearance(pos, tree_specs, drone_radius: float, drone_height: float):
        if len(tree_specs) == 0:
            return np.inf

        pos = np.asarray(pos, dtype=np.float32)
        best = np.inf
        for spec in tree_specs:
            height = float(spec["height"])
            if pos[2] > height + drone_height:
                continue
            center_xy = spec["xy"]
            radial_clearance = float(np.linalg.norm(pos[:2] - center_xy) - spec["radius"] - drone_radius)
            if radial_clearance < best:
                best = radial_clearance

        return best if np.isfinite(best) else np.inf

    def _is_tree_placement_valid(self, *, xy, radius: float, start_pos, goal_pos, tree_specs, corridor_half_width: float, protect_corridor: bool):
        start_xy = start_pos[:2]
        goal_xy = goal_pos[:2]

        start_margin = 2.5 * radius + max(0.25, corridor_half_width)
        if np.linalg.norm(xy - start_xy) < start_margin:
            return False

        goal_margin = 2.5 * radius + max(0.25, corridor_half_width)
        if np.linalg.norm(xy - goal_xy) < goal_margin:
            return False

        if protect_corridor and distance_point_to_segment_2d(xy, start_xy, goal_xy) < (radius + corridor_half_width):
            return False

        for spec in tree_specs:
            min_sep = radius + spec["radius"] + self.config.min_tree_separation
            if np.linalg.norm(xy - spec["xy"]) < min_sep:
                return False

        return True

    def _sample_tree_xy(
        self,
        *,
        rng,
        radius: float,
        start_pos,
        goal_pos,
        corridor_half_width: float,
        protect_corridor: bool,
        corridor_edge_tree_fraction: float,
        centerline_tree_fraction: float,
    ):
        if protect_corridor and rng.random() < corridor_edge_tree_fraction:
            xy = self._sample_near_route_xy(
                rng=rng,
                radius=radius,
                start_pos=start_pos,
                goal_pos=goal_pos,
                min_offset=corridor_half_width + radius + 0.05,
                max_offset=corridor_half_width + radius + self.config.centerline_band_width,
            )
            if xy is not None:
                return xy

        if (not protect_corridor) and rng.random() < centerline_tree_fraction:
            xy = self._sample_near_route_xy(
                rng=rng,
                radius=radius,
                start_pos=start_pos,
                goal_pos=goal_pos,
                min_offset=0.0,
                max_offset=self.config.centerline_band_width,
            )
            if xy is not None:
                return xy

        return rng.uniform(
            low=-self.config.forest_half_extent,
            high=self.config.forest_half_extent,
            size=2,
        ).astype(np.float32)

    def _sample_near_route_xy(self, *, rng, radius: float, start_pos, goal_pos, min_offset: float, max_offset: float):
        start_xy = np.asarray(start_pos[:2], dtype=np.float32)
        goal_xy = np.asarray(goal_pos[:2], dtype=np.float32)
        route_dir, perp_dir = route_direction_xy(start_xy, goal_xy)
        if np.linalg.norm(route_dir) < 1e-8:
            return None

        for _ in range(24):
            base_xy = self._sample_point_along_route_inside_forest(rng=rng, start_xy=start_xy, goal_xy=goal_xy)
            if base_xy is None:
                return None

            if min_offset <= 0.0:
                offset_mag = float(rng.uniform(-max_offset, max_offset))
            else:
                offset_mag = float(rng.uniform(min_offset, max_offset))
                offset_mag *= -1.0 if rng.random() < 0.5 else 1.0

            xy = base_xy + perp_dir * offset_mag
            if np.all(np.abs(xy) <= self.config.forest_half_extent - max(radius, 0.05)):
                return xy.astype(np.float32)

        return None

    def _sample_point_along_route_inside_forest(self, *, rng, start_xy, goal_xy):
        for _ in range(32):
            t = float(rng.uniform(0.0, 1.0))
            point = start_xy + t * (goal_xy - start_xy)
            if np.all(np.abs(point) <= self.config.forest_half_extent - 0.05):
                return point.astype(np.float32)
        return None
