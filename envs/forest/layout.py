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
    route_blocking_tree: bool = True
    route_tree_fraction: float = 0.5
    route_blocking_tree_count: int = 1
    route_tree_lateral_range: float = 0.55
    route_tree_layout: str = "random"


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
        target_tree_count = self.config.num_trees

        route_tree_specs = self._create_route_blocking_tree_specs(
            rng=rng,
            start_pos=start_pos,
            goal_pos=goal_pos,
            corridor_half_width=corridor_half_width,
        )
        tree_specs.extend(route_tree_specs)
        target_tree_count += len(route_tree_specs)

        attempts_per_tree = 250 if self.config.route_tree_layout == "fixed_safe_five" else 40
        max_attempts = max(200, target_tree_count * attempts_per_tree)

        while len(tree_specs) < target_tree_count and attempts < max_attempts:
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

        if len(tree_specs) != target_tree_count:
            raise RuntimeError(
                f"Unable to generate requested forest layout: placed {len(tree_specs)} "
                f"of {target_tree_count} trees after {max_attempts} attempts"
            )
        return tree_specs

    def _create_route_blocking_tree_specs(self, *, rng, start_pos, goal_pos, corridor_half_width: float):
        if not self.config.route_blocking_tree:
            return []

        start_xy = np.asarray(start_pos[:2], dtype=np.float32)
        goal_xy = np.asarray(goal_pos[:2], dtype=np.float32)
        route = goal_xy - start_xy
        if np.linalg.norm(route) < 1e-8:
            return []

        count = max(0, int(self.config.route_blocking_tree_count))
        if self.config.route_tree_layout == "fixed_safe_five":
            if count != 5:
                raise ValueError("fixed_safe_five requires route_blocking_tree_count=5")
            fixed = (
                (0.243, 0.15),
                (0.357, -0.55),
                (0.493, 0.45),
                (0.643, -0.35),
                (0.771, 0.25),
            )
            radius = 0.16
            height = 2.0
            return [
                {
                    "xy": (start_xy + fraction * route + lateral * route_direction_xy(start_xy, goal_xy)[1]).astype(np.float32),
                    "radius": radius,
                    "height": height,
                    "route_blocking": True,
                    "route_fraction": fraction,
                    "route_lateral_offset": lateral,
                }
                for fraction, lateral in fixed
            ]
        route_specs = []
        route_dir, perp_dir = route_direction_xy(start_xy, goal_xy)
        attempts = 0
        max_attempts = max(100, count * 100)
        while len(route_specs) < count and attempts < max_attempts:
            attempts += 1
            radius = float(rng.uniform(*self.config.tree_radius_range))
            height = float(rng.uniform(*self.config.tree_height_range))
            if count == 1:
                fraction = float(np.clip(self.config.route_tree_fraction, 0.05, 0.95))
                lateral_offset = 0.0
            else:
                fraction = float(rng.uniform(0.08, 0.92))
                lateral_offset = float(
                    rng.uniform(-self.config.route_tree_lateral_range, self.config.route_tree_lateral_range)
                )
            xy = (start_xy + fraction * route + lateral_offset * perp_dir).astype(np.float32)
            if np.any(np.abs(xy) > self.config.forest_half_extent - max(radius, 0.05)):
                continue
            if not self._is_tree_placement_valid(
                xy=xy,
                radius=radius,
                start_pos=start_pos,
                goal_pos=goal_pos,
                tree_specs=route_specs,
                corridor_half_width=corridor_half_width,
                protect_corridor=False,
            ):
                continue
            route_specs.append(
                {
                    "xy": xy,
                    "radius": radius,
                    "height": height,
                    "route_blocking": True,
                    "route_fraction": fraction,
                    "route_lateral_offset": lateral_offset,
                }
            )
        return route_specs

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

        if self.config.route_tree_layout == "fixed_safe_five":
            route = goal_xy - start_xy
            _, perp_dir = route_direction_xy(start_xy, goal_xy)
            safe_path = (
                start_xy,
                start_xy + 0.143 * route + 1.1 * perp_dir,
                start_xy + 0.857 * route + 1.1 * perp_dir,
                goal_xy,
            )
            required_centerline_distance = radius + 0.06 + 0.35
            if any(
                distance_point_to_segment_2d(xy, begin, end) < required_centerline_distance
                for begin, end in zip(safe_path, safe_path[1:])
            ):
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
