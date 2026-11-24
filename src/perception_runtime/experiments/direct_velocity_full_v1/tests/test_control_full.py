import numpy as np
import pytest

from experiments.direct_velocity_full_v1.contracts import ModelParameters, ControlSpec, VelocityCommand, ControlState
from experiments.direct_velocity_full_v1.controls.velocity_tracking import VelocityTrackingController
from experiments.direct_velocity_full_v1.testing.control_fixture import FullControlFixture


def advance(fixture, controller, count, command=(0, 0, 0)):
    records = []
    for k in range(count):
        rpm, diagnostic = controller.update(VelocityCommand(k//8, k, command) if k % 8 == 0 else None,
                                             fixture.motion_state())
        motor = fixture.motor_action_codec.rpm_to_normalized_action(rpm).astype(np.float32)
        fixture.step(motor.reshape(1, 4))
        records.append(np.r_[fixture.vel[0], fixture.rpy[0], fixture._last_motor_rpm[0]])
    return np.array(records)


def test_P01_exact_120_240_rotor_physics_and_command_hold():
    fixture = FullControlFixture()
    try:
        fixture.reset(seed=0)
        controller = VelocityTrackingController()
        controller.reset(fixture.motion_state())
        result = advance(fixture, controller, 8, [.8, 0, 0])
        assert controller.tick == 8
        assert fixture.physics_ticks == fixture.step_counter == 16
        assert fixture.physics_ticks/240 == pytest.approx(1/15)
        assert fixture.vel[0][0] > 0
        assert np.isfinite(result).all()
    finally:
        fixture.close()


def test_D01_force_applied_for_every_physics_substep():
    fixture = FullControlFixture()
    try:
        fixture.reset(seed=0)
        fixture.disturbances = [dict(start_s=0., duration_s=.15, force_world_n=[.054, -.036, .018])]
        controller = VelocityTrackingController()
        controller.reset(fixture.motion_state())
        advance(fixture, controller, 40)
        assert fixture.applied_disturbance_substeps == 36
        assert fixture.physics_ticks == 80
    finally:
        fixture.close()


def test_R01_saturated_controller_and_used_physics_reset_match_fresh():
    fixtures = [FullControlFixture(), FullControlFixture()]
    controllers = [VelocityTrackingController(), VelocityTrackingController()]
    try:
        fixtures[0].reset(seed=0)
        controllers[0].reset(fixtures[0].motion_state())
        advance(fixtures[0], controllers[0], 120, [.8, 0, .3])
        for tick in range(200):
            motion = ControlState([0, 0, 0, 1], [-5, 0, -5], [0, 0, 0])
            controllers[0].update(None, motion)
        controllers[0].finish_episode()
        results = []
        for fixture, controller in zip(fixtures, controllers):
            fixture.reset(seed=1)
            assert fixture._last_motor_rpm is None
            controller.reset(fixture.motion_state())
            assert controller.command is None and controller.tick == 0
            np.testing.assert_array_equal(controller.pi.integral, [0, 0, 0])
            np.testing.assert_array_equal(controller.adapter.integral, [0, 0, 0])
            results.append(advance(fixture, controller, 120, [.4, -.1, .1]))
        np.testing.assert_allclose(results[0], results[1], rtol=1e-6, atol=1e-6)
    finally:
        for fixture in fixtures:
            fixture.close()


def test_U04_translated_aircraft_has_same_velocity_control_outputs():
    fixtures = [FullControlFixture(position=position) for position in ([0, 0, 1], [5, -3, 1])]
    results = []
    try:
        for fixture in fixtures:
            fixture.reset(seed=0)
            controller = VelocityTrackingController()
            controller.reset(fixture.motion_state())
            results.append(advance(fixture, controller, 120, [.3, .2, 0]))
        np.testing.assert_allclose(results[0], results[1], rtol=1e-6, atol=1e-6)
    finally:
        for fixture in fixtures:
            fixture.close()


def test_early_done_stops_before_remaining_control_ticks():
    fixture = FullControlFixture()
    try:
        fixture.reset(seed=0)
        fixture.TARGET_POS = fixture.pos[0].copy()
        controller = VelocityTrackingController()
        controller.reset(fixture.motion_state())
        for k in range(8):
            rpm, _ = controller.update(VelocityCommand(0, 0, [0, 0, 0]) if k == 0 else None,
                                       fixture.motion_state())
            motor = fixture.motor_action_codec.rpm_to_normalized_action(rpm).astype(np.float32)
            _, _, terminated, truncated, _ = fixture.step(motor.reshape(1, 4))
            if terminated or truncated:
                controller.finish_episode()
                break
        assert controller.tick == 1 and fixture.physics_ticks == 2
        with pytest.raises(RuntimeError):
            controller.update(None, fixture.motion_state())
        assert fixture.physics_ticks == 2
    finally:
        fixture.close()
