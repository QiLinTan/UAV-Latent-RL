import unittest
from pathlib import Path

import numpy as np

from utils.gym_pybullet_compat import ensure_gym_pybullet_envs_compat

ensure_gym_pybullet_envs_compat()

from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from envs.ForestAviary import CustomForestAviary
from envs.learned_upper_dslpid import LearnedUpperDSLPIDEnv
from envs.map_atlas import MapCatalog, MapSpec
from models.reference_packet import ReferencePacket


CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "map_atlas" / "catalog_v1.json"


class MapAtlasTest(unittest.TestCase):
    def setUp(self):
        self.catalog = MapCatalog(CATALOG_PATH)

    def test_catalog_has_disjoint_training_validation_and_test_entries(self):
        self.assertEqual(
            self.catalog.ids(split="train"),
            (
                "procedural_forest_multitree_train_v2",
                "procedural_forest_train_v1",
            ),
        )
        self.assertEqual(
            self.catalog.ids(split="validation"),
            ("procedural_forest_validation_v1", "subt_tunnel_proxy_v1"),
        )
        self.assertEqual(
            self.catalog.ids(split="test"),
            ("flightbench_maze_proxy_v1", "procedural_forest_mirror_probe_v1"),
        )
        for map_id in self.catalog.ids():
            report = self.catalog.get(map_id).validate_route()
            self.assertTrue(report.endpoints_valid, map_id)
            self.assertIsNot(report.reachable, False, map_id)

    def test_blocked_primitive_map_is_rejected_before_pybullet_is_started(self):
        spec = MapSpec.from_dict(
            {
                "map_id": "blocked",
                "source": "unit_test",
                "scenario": "test",
                "split": "test",
                "start_pos": [-1.0, 0.0, 1.0],
                "target_pos": [1.0, 0.0, 1.0],
                "xy_half_extent": 1.5,
                "obstacle_sensor_range": 1.0,
                "obstacles": [
                    {
                        "kind": "box",
                        "name": "full_barrier",
                        "position": [0.0, 0.0, 1.0],
                        "half_extents": [0.2, 1.6, 1.0],
                    }
                ],
            },
            root=CATALOG_PATH.parent,
        )
        report = spec.validate_route()
        self.assertTrue(report.endpoints_valid)
        self.assertFalse(report.reachable)
        with self.assertRaisesRegex(ValueError, "cannot be used"):
            CustomForestAviary(
                obs=ObservationType.KIN,
                act=ActionType.RPM,
                pyb_freq=240,
                ctrl_freq=120,
                gui=False,
                map_spec=spec,
            )

    def test_static_map_preserves_reference_packet_execution_contract(self):
        spec = self.catalog.get("subt_tunnel_proxy_v1")
        base_env = CustomForestAviary(
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            pyb_freq=240,
            ctrl_freq=120,
            gui=False,
            map_spec=spec,
        )
        env = LearnedUpperDSLPIDEnv(base_env, upper_control_interval=8)
        try:
            observation, reset_info = env.reset(seed=3)
            self.assertEqual(observation.shape, (1, 29))
            self.assertEqual(reset_info["map_id"], "subt_tunnel_proxy_v1")
            self.assertEqual(reset_info["num_obstacles"], len(spec.obstacles))
            self.assertTrue(reset_info["map_route_reachable"])

            next_observation, reward, terminated, truncated, info = env.step(
                np.array([[0.2, 0.0, 0.0]], dtype=np.float32)
            )
            self.assertEqual(next_observation.shape, (1, 29))
            self.assertTrue(np.isfinite(reward))
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertIsInstance(env.last_reference_packet, ReferencePacket)
            self.assertTrue(info["reference_packet_valid"])
            self.assertEqual(info["map_id"], spec.map_id)
            self.assertEqual(info["map_split"], "validation")
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
