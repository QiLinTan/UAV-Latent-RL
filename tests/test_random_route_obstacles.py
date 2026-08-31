import unittest

import numpy as np

from envs.forest.geometry import distance_point_to_segment_2d
from envs.forest.layout import ForestLayoutConfig, ForestLayoutGenerator


class RandomRouteObstacleTest(unittest.TestCase):
    def _generator(self):
        return ForestLayoutGenerator(
            ForestLayoutConfig(
                forest_half_extent=3.0,
                num_trees=24,
                tree_radius_range=(0.10, 0.22),
                tree_height_range=(1.2, 2.4),
                min_tree_separation=0.45,
                centerline_band_width=0.40,
                route_blocking_tree=True,
                route_blocking_tree_count=5,
                route_tree_lateral_range=0.55,
            )
        )

    def test_five_random_route_obstacles_are_generated_without_overlap(self):
        layouts = []
        for seed in range(10):
            specs = self._generator().generate(
                rng=np.random.default_rng(seed),
                start_pos=[-3.5, 0.0, 1.0],
                goal_pos=[3.5, 0.0, 1.0],
                corridor_half_width=1.35,
                protect_corridor=True,
                corridor_edge_tree_fraction=0.0,
                centerline_tree_fraction=0.0,
            )
            route_specs = [spec for spec in specs if spec.get("route_blocking")]
            self.assertEqual(len(route_specs), 5)
            self.assertEqual(len(specs), 29)
            for index, first in enumerate(route_specs):
                for second in route_specs[index + 1 :]:
                    required = first["radius"] + second["radius"] + 0.45
                    self.assertGreaterEqual(np.linalg.norm(first["xy"] - second["xy"]), required)
            layouts.append(tuple(tuple(np.round(spec["xy"], 3)) for spec in route_specs))
        self.assertGreater(len(set(layouts)), 1)

    def test_fixed_five_layout_has_a_continuous_safe_route(self):
        generator = ForestLayoutGenerator(
            ForestLayoutConfig(
                forest_half_extent=3.0,
                num_trees=0,
                tree_radius_range=(0.10, 0.22),
                tree_height_range=(1.2, 2.4),
                min_tree_separation=0.45,
                centerline_band_width=0.40,
                route_blocking_tree=True,
                route_blocking_tree_count=5,
                route_tree_layout="fixed_safe_five",
            )
        )
        specs = generator.generate(
            rng=np.random.default_rng(0),
            start_pos=[-3.5, 0.0, 1.0],
            goal_pos=[3.5, 0.0, 1.0],
            corridor_half_width=1.35,
            protect_corridor=True,
            corridor_edge_tree_fraction=0.0,
            centerline_tree_fraction=0.0,
        )
        self.assertEqual(len(specs), 5)
        safe_polyline = (
            np.array([-3.5, 0.0]),
            np.array([-2.5, 1.1]),
            np.array([2.5, 1.1]),
            np.array([3.5, 0.0]),
        )
        drone_radius = 0.06
        required_clearance = 0.35
        for spec in specs:
            centerline_distance = min(
                distance_point_to_segment_2d(spec["xy"], start, end)
                for start, end in zip(safe_polyline, safe_polyline[1:])
            )
            clearance = centerline_distance - spec["radius"] - drone_radius
            self.assertGreaterEqual(clearance, required_clearance)

    def test_24_background_trees_preserve_the_verified_safe_route(self):
        generator = ForestLayoutGenerator(
            ForestLayoutConfig(
                forest_half_extent=3.0,
                num_trees=24,
                tree_radius_range=(0.10, 0.22),
                tree_height_range=(1.2, 2.4),
                min_tree_separation=0.45,
                centerline_band_width=0.40,
                route_blocking_tree=True,
                route_blocking_tree_count=5,
                route_tree_layout="fixed_safe_five",
            )
        )
        safe_polyline = (
            np.array([-3.5, 0.0]),
            np.array([-2.5, 1.1]),
            np.array([2.5, 1.1]),
            np.array([3.5, 0.0]),
        )
        for seed in range(20):
            specs = generator.generate(
                rng=np.random.default_rng(seed),
                start_pos=[-3.5, 0.0, 1.0],
                goal_pos=[3.5, 0.0, 1.0],
                corridor_half_width=1.35,
                protect_corridor=True,
                corridor_edge_tree_fraction=0.0,
                centerline_tree_fraction=0.0,
            )
            self.assertEqual(len(specs), 29)
            for spec in specs:
                centerline_distance = min(
                    distance_point_to_segment_2d(spec["xy"], start, end)
                    for start, end in zip(safe_polyline, safe_polyline[1:])
                )
                clearance = centerline_distance - spec["radius"] - 0.06
                self.assertGreaterEqual(clearance, 0.35)


if __name__ == "__main__":
    unittest.main()
