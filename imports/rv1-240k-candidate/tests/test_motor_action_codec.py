import unittest

import numpy as np

from models.motor_action_codec import (
    ASYMMETRIC_RPM,
    ASYMMETRIC_THRUST,
    MotorActionCodec,
    MotorPhysicalLimits,
    PhysicalMotorConstraint,
)


class MotorActionCodecTest(unittest.TestCase):
    def setUp(self):
        self.limits = MotorPhysicalLimits(
            min_rpm=0.0,
            hover_rpm=14468.4,
            max_rpm=21702.6,
            kf=3.16e-10,
        )

    def test_hover_and_physical_endpoints_have_expected_actions(self):
        for mode in (ASYMMETRIC_RPM, ASYMMETRIC_THRUST):
            codec = MotorActionCodec(self.limits, mode=mode)
            actions = codec.rpm_to_normalized_action(
                [self.limits.min_rpm, self.limits.hover_rpm, self.limits.max_rpm]
            )
            np.testing.assert_allclose(actions, [-1.0, 0.0, 1.0], atol=1e-12)

    def test_rpm_round_trip_is_numerically_reversible(self):
        rng = np.random.default_rng(7)
        rpm = rng.uniform(self.limits.min_rpm, self.limits.max_rpm, size=(2048, 4))
        for mode in (ASYMMETRIC_RPM, ASYMMETRIC_THRUST):
            codec = MotorActionCodec(self.limits, mode=mode)
            reconstructed = codec.normalized_action_to_rpm(
                codec.rpm_to_normalized_action(rpm)
            )
            np.testing.assert_allclose(reconstructed, rpm, rtol=1e-12, atol=1e-9)

    def test_motor_thrust_round_trip_is_numerically_reversible(self):
        codec = MotorActionCodec(self.limits, mode=ASYMMETRIC_THRUST)
        thrust = np.linspace(
            self.limits.min_thrust,
            self.limits.max_thrust,
            4096,
        )
        reconstructed = codec.normalized_action_to_motor_thrust(
            codec.motor_thrust_to_normalized_action(thrust)
        )
        np.testing.assert_allclose(reconstructed, thrust, rtol=1e-12, atol=1e-15)

    def test_physical_constraint_does_not_project_collective_or_differential(self):
        constraint = PhysicalMotorConstraint(self.limits)
        valid = np.array([1000.0, 7000.0, 15000.0, 21000.0])
        np.testing.assert_allclose(constraint(valid), valid)
        clipped = constraint([-10.0, 100.0, 20000.0, 30000.0])
        np.testing.assert_allclose(
            clipped,
            [self.limits.min_rpm, 100.0, 20000.0, self.limits.max_rpm],
        )


if __name__ == "__main__":
    unittest.main()
