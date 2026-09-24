import unittest

from trainers.collision_diagnostics import classify_collision_window


def _sample(*, command, actual, clearance, action=None, ray=0.5):
    return {
        "cmd_velocity": command,
        "actual_velocity": actual,
        "upper_action": command if action is None else action,
        "clearance": clearance,
        "min_range_fraction": ray,
    }


class CollisionWindowClassificationTest(unittest.TestCase):
    def test_a_identifies_unreduced_forward_command(self):
        samples = [
            _sample(command=(0.6, 0.0, 0.0), actual=(0.58, 0.0, 0.0), clearance=value)
            for value in (1.2, 0.8, 0.3, 0.0)
        ]
        result = classify_collision_window(samples)
        self.assertEqual(result["code"], "A")

    def test_b_identifies_brake_command_not_realized(self):
        samples = [
            _sample(command=command, actual=actual, clearance=clearance)
            for command, actual, clearance in (
                ((0.6, 0.0, 0.0), (0.58, 0.0, 0.0), 1.2),
                ((0.3, 0.0, 0.0), (0.54, 0.0, 0.0), 0.7),
                ((0.0, 0.0, 0.0), (0.35, 0.0, 0.0), 0.0),
            )
        ]
        result = classify_collision_window(samples)
        self.assertEqual(result["code"], "B")

    def test_d_identifies_absent_ray_warning(self):
        samples = [
            _sample(command=(0.2, 0.0, 0.0), actual=(0.2, 0.0, 0.0), clearance=value, ray=1.0)
            for value in (0.6, 0.2, 0.0)
        ]
        result = classify_collision_window(samples)
        self.assertEqual(result["code"], "D")


if __name__ == "__main__":
    unittest.main()
