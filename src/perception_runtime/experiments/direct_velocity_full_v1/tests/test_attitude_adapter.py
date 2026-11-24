import numpy as np
import pytest
import pybullet as p
from scipy.spatial.transform import Rotation
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

from experiments.direct_velocity_full_v1.contracts import (
    ControlState, VelocityCommand, ModelParameters, ControlSpec, collective_force_to_pwm, force_to_attitude,
)
from experiments.direct_velocity_full_v1.controls.velocity_control_adapter import DSLAttitudeAdapter
from experiments.direct_velocity_full_v1.controls.velocity_tracking import VelocityTrackingController


def state(rpy=(0, 0, 0), velocity=(0, 0, 0)):
    return ControlState(p.getQuaternionFromEuler(rpy), velocity, [0, 0, 0])


def test_isolated_formula_matches_original_private_function():
    rng = np.random.default_rng(23)
    model = ModelParameters()
    for _ in range(100):
        initial_rpy = rng.uniform(-.005, .005, 3)
        current = state(initial_rpy)
        adapter = DSLAttitudeAdapter(model)
        adapter.reset(current)
        original = DSLPIDControl(DroneModel.CF2X)
        original.last_rpy = np.asarray(p.getEulerFromQuaternion(current.q_world_from_body_xyzw))
        force = np.r_[rng.uniform(-.0005, .0005, 2), model.weight_n]
        _, euler = force_to_attitude(force, adapter.yaw_target)
        from experiments.direct_velocity_full_v1.contracts import body_to_world
        thrust = force @ body_to_world(current.q_world_from_body_xyzw)[:, 2]
        expected = original._dslPIDAttitudeControl(1/120, collective_force_to_pwm(thrust, model),
                                                   current.q_world_from_body_xyzw, euler, np.zeros(3))
        actual, diagnostic = adapter.update(force, current)
        assert not diagnostic['allocation_saturated']
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-12)
        np.testing.assert_allclose(adapter.integral, original.integral_rpy_e, atol=1e-14)


@pytest.mark.parametrize('yaw', [0, np.pi/2, np.pi, -np.pi+1e-4])
def test_nonzero_initial_attitude_has_no_false_derivative(yaw):
    adapter = DSLAttitudeAdapter(ModelParameters())
    current = state((.1, -.1, yaw))
    adapter.reset(current)
    rpm, diagnostic = adapter.update([0, 0, .2646], current)
    np.testing.assert_array_equal(diagnostic['measured_euler_rates'], [0, 0, 0])
    assert np.isfinite(rpm).all()


def test_yaw_wrap_difference_and_gravity_geometry():
    adapter = DSLAttitudeAdapter(ModelParameters())
    adapter.reset(state((0, 0, np.pi-1e-4)))
    _, diagnostic = adapter.update([0, 0, .2646], state((0, 0, -np.pi+1e-4)))
    assert diagnostic['measured_euler_rates'][2] == pytest.approx(2e-4 * 120, abs=1e-10)
    assert np.linalg.norm(diagnostic['measured_euler_rates']) < .1
    np.testing.assert_allclose(diagnostic['target_force_direction'], [0, 0, 1], atol=1e-14)


def test_yaw_integral_does_not_accumulate_against_saturated_mixer():
    adapter = DSLAttitudeAdapter(ModelParameters())
    adapter.reset(state())
    for _ in range(300):
        rpm, diagnostic = adapter.update([0, 0, .2646], state((0, 0, .8)))
    assert diagnostic['allocation_saturated']
    assert diagnostic['attitude_integrator_blocked'][2]
    assert adapter.integral[2] == 0
    assert np.all((rpm >= 0) & (rpm <= adapter.model.max_rpm))


def test_pwm_and_physical_rpm_limit_and_zero_velocity_hover():
    model = ModelParameters()
    adapter = DSLAttitudeAdapter(model)
    adapter.reset(state())
    rpm, diagnostic = adapter.update([0, 0, 10], state())
    assert diagnostic['pwm_saturation_mask'].all()
    assert np.all(rpm <= model.max_rpm)
    adapter.reset(state())
    rpm, _ = adapter.update([0, 0, model.weight_n], state())
    np.testing.assert_allclose(rpm, model.hover_rpm, atol=1e-9)
    assert np.all(rpm > 0)
    # Numerical boundary test only: deliberately tighter physical envelope.
    from dataclasses import replace
    tight_model = replace(model, thrust_weight_ratio=1.1)
    tight = DSLAttitudeAdapter(tight_model)
    tight.reset(state())
    limited, diagnostic = tight.update([0, 0, 10], state())
    assert diagnostic['physical_rpm_saturation_mask'].all()
    assert np.all(limited <= tight_model.max_rpm)


def test_U01_atomic_validation_hold_reset_and_lifecycle():
    controller = VelocityTrackingController(ModelParameters())
    with pytest.raises(RuntimeError):
        controller.update(None, state())
    controller.reset(state((.05, -.05, 1.)))
    for dt in (0, -1, np.nan, .01):
        before = controller.snapshot()
        with pytest.raises(ValueError):
            controller.update(None, state(), dt)
        assert controller.snapshot() == before
    controller.update(VelocityCommand(1, 0, [.4, 0, 0]), state())
    before = controller.snapshot()
    for invalid in (VelocityCommand(0, 0, [0, 0, 0]), VelocityCommand(1, 0, [.5, 0, 0]), VelocityCommand(2, 2, [0, 0, 0])):
        with pytest.raises(ValueError):
            controller.update(invalid, state())
        assert controller.snapshot() == before
    _, diagnostic = controller.update(None, state())
    np.testing.assert_array_equal(diagnostic['command_applied_world'], [.4, 0, 0])
    controller.finish_episode()
    assert not controller.active
    assert not np.any(controller.pi.integral)
    with pytest.raises(RuntimeError):
        controller.update(None, state())


def test_atomic_adapter_error_rolls_back_complete_controller():
    class FaultAdapter(DSLAttitudeAdapter):
        def update(self, *args, **kwargs):
            super().update(*args, **kwargs)
            raise ValueError('injected allocation error')
    model = ModelParameters()
    controller = VelocityTrackingController(model, attitude_adapter=FaultAdapter(model))
    controller.reset(state())
    before = controller.snapshot()
    with pytest.raises(ValueError):
        controller.update(VelocityCommand(0, 0, [.5, 0, 0]), state())
    assert controller.snapshot() == before


def test_U04_motion_only_control_outputs_identical():
    a, b = VelocityTrackingController(ModelParameters()), VelocityTrackingController(ModelParameters())
    a.reset(state())
    b.reset(state())
    command = VelocityCommand(0, 0, [.4, -.2, .1])
    # Different world positions cannot be supplied: they are absent from API.
    for tick in range(120):
        ar, _ = a.update(command if tick == 0 else None, state())
        br, _ = b.update(command if tick == 0 else None, state())
        np.testing.assert_array_equal(ar, br)
