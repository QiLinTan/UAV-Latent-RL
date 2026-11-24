"""Focused QA for Suite D's geometry and scoring (no checkpoint or GPU)."""
import unittest
from types import SimpleNamespace

import numpy as np

from geometry import blocked, extents, generate, solve
from metrics import clearance, milestones, obstacle_match, terminal_failure


class SuiteDTests(unittest.TestCase):
    def test_timeout_uses_advanced_physics_clock(self):
        from contract import reason_for

        base = SimpleNamespace(TARGET_POS=np.array([3., 0., 1.]), suite_d_closed=True,
                               physics_ticks=2400, step_counter=2398, suite_d_timeout_ticks=2400)
        state = np.zeros(20)
        state[2] = 1.
        self.assertEqual(reason_for(base, state, collision=False), "timeout")
        base.physics_ticks -= 2
        self.assertEqual(reason_for(base, state, collision=False), "running")
        self.assertEqual(reason_for(base, state, collision=True), "collision")

    def test_scenarios_and_mirror(self):
        for topology in ("D1", "D2", "D3", "D4"):
            for index in (0, 19):
                record, details, _, closed = generate(topology, index)
                graph = solve(record)
                lower, upper = extents(record)
                self.assertFalse(blocked(graph.route[:-1], graph.route[1:], lower, upper, .162).any())
                self.assertGreater(graph.length, 0)
                if topology == "D2":
                    self.assertGreaterEqual(solve(record, aperture_only=True).length + 1e-8, graph.length)
                if topology == "D4":
                    mirror = solve(generate(topology, index ^ 1)[0])
                    self.assertAlmostEqual(graph.length, mirror.length, places=8)
                    self.assertNotEqual(details["target_arm"], generate(topology, index ^ 1)[1]["target_arm"])
                self.assertEqual(closed, topology in ("D3", "D4"))

    def test_window_must_cross_front_and_rear(self):
        record, details, _, _ = generate("D2", 0)
        case = dict(topology="D2", details=details, map=record)
        x = details["x"]
        y, z = np.mean(details["y_window"]), np.mean(details["z_window"])
        straight = np.array([[x - .3, y, z], [x, y, z], [x + .3, y, z]])
        artificial = np.array([[x - .3, 2., z], [x, 2., z], [x + .10, y, z], [x + .3, y, z]])
        self.assertTrue(milestones(case, straight, np.zeros(3), "success")["through_then_success"])
        self.assertFalse(milestones(case, artificial, np.zeros(4), "success")["through_window"])

    def test_raw_hit_requires_matching_body_and_location(self):
        scan = dict(body=np.array([7, 4]), xyz=np.array([[1, 0, 0], [1, .5, 0]]),
                    origin=np.zeros(3), directions=np.array([[1, 0, 0], [1, .5, 0]]) / [[1], [np.sqrt(1.25)]])
        self.assertEqual(obstacle_match([scan] * 3, dict(id=7, point=[1, 0, 0])), [True] * 3)
        self.assertEqual(obstacle_match([scan] * 3, dict(id=4, point=[1, 0, 0])), [False] * 3)

    def test_failure_priority_and_unclassified(self):
        record, details, _, _ = generate("D1", 0)
        case = dict(topology="D1", details=details, map=record)
        positions = np.array([[details["x"], 0., details["z"]]] * 7)
        velocity = np.tile([.7, 0, 0], (7, 1))
        contact = dict(name="thin_beam", normal=[1, 0, 0], point=positions[-1].tolist())
        arguments = (positions, velocity, np.zeros(7), np.zeros(7))
        bucket, flags = terminal_failure(case, "collision", *arguments, [contact], [[False] * 3], 4)
        self.assertEqual(bucket, "F1")
        self.assertIn("F3", flags)
        self.assertEqual(terminal_failure(case, "xy_bound", *arguments, [], [], 4)[0], "unclassified")
        self.assertLess(clearance(record, positions)[0], 0)


if __name__ == "__main__":
    unittest.main()
