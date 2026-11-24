import inspect

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from experiments.direct_velocity_full_v1.contracts import (
    ControlState, VelocityCommand, ModelParameters, ControlSpec, body_to_world,
    yaw_rotation, world_to_local, local_to_world, body_angular_velocity,
    normalized_action_to_world, project_world_velocity, force_to_attitude,
    collective_force_to_pwm, pwm_to_collective_force, rpm_to_collective_force,
    quaternion, wrap_pi,
)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_U01_finite_inputs(bad):
    for function in (
        lambda: ControlState([0, 0, 0, 1], [bad, 0, 0], [0, 0, 0]),
        lambda: VelocityCommand(0, 0, [bad, 0, 0]),
        lambda: project_world_velocity([bad, 0, 0]),
        lambda: collective_force_to_pwm(bad, ModelParameters()),
    ):
        with pytest.raises(ValueError):
            function()


def test_U01_shapes_quaternion_ids_immutability():
    for q in ([0, 0, 0, 0], [0, 0, 0, 2], [[0, 0, 0, 1]], [0, 0, 1]):
        with pytest.raises(ValueError):
            quaternion(q)
    for sid in (-1, .5, True):
        with pytest.raises(ValueError):
            VelocityCommand(sid, 0, [0, 0, 0])
    with pytest.raises(ValueError):
        project_world_velocity([[0, 0, 0]])
    state = ControlState([0, 0, 0, 1], [0, 0, 0], [0, 0, 0])
    with pytest.raises(ValueError):
        state.velocity_world[0] = 1


def test_U02_disk_and_vertical_projection():
    np.testing.assert_allclose(normalized_action_to_world([1, 1, 1], 0), [.8 / np.sqrt(2), .8 / np.sqrt(2), .3])
    rng = np.random.default_rng(11)
    for value in rng.normal(size=(1000, 3)) * 10:
        result = project_world_velocity(value)
        assert np.linalg.norm(result[:2]) <= .8 + 1e-12
        assert abs(result[2]) <= .3


def test_U03_rotation_gravity_and_angular_velocity():
    q = Rotation.from_euler('xyz', [.2, -.1, 1.2]).as_quat()
    matrix = body_to_world(q)
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(body_to_world(-q), matrix)
    np.testing.assert_allclose(body_angular_velocity(matrix @ [1, 2, 3], q), [1, 2, 3])
    np.testing.assert_allclose(np.linalg.norm(matrix.T @ [0, 0, -1]), 1)
    for yaw in (0, np.pi / 2, np.pi, -np.pi + 1e-6):
        np.testing.assert_allclose(world_to_local(local_to_world([.2, -.3, .1], yaw), yaw), [.2, -.3, .1])
    np.testing.assert_allclose(wrap_pi([-2*np.pi+1e-4, 2*np.pi-1e-4]), [1e-4, -1e-4], atol=1e-12)


def test_U03_force_pwm_rpm_units_and_geometry():
    model = ModelParameters.from_installed()
    assert model.mass_kg == .027
    assert model.weight_n == pytest.approx(.2646)
    for force in np.linspace(.01, model.max_force_n, 1000):
        pwm = collective_force_to_pwm(force, model)
        assert pwm_to_collective_force(pwm, model) == pytest.approx(force, rel=1e-12)
        rpm = model.pwm_scale * pwm + model.pwm_offset
        assert rpm_to_collective_force(np.repeat(rpm, 4), model) == pytest.approx(force, rel=1e-12)
    for force in ([0, 0, model.weight_n], [.08, .02, model.weight_n], [-.04, -.07, .3]):
        for yaw in (0, np.pi/2, np.pi):
            matrix, euler = force_to_attitude(force, yaw)
            np.testing.assert_allclose(matrix[:, 2], np.asarray(force) / np.linalg.norm(force), atol=1e-14)
            np.testing.assert_allclose(Rotation.from_euler('XYZ', euler).as_matrix(), matrix, atol=1e-14)
            assert np.linalg.det(matrix) == pytest.approx(1)


def test_U04_control_data_contract_has_no_position_reference():
    assert set(inspect.signature(ControlState).parameters) == {
        'q_world_from_body_xyzw', 'velocity_world', 'angular_velocity_world'}
    assert set(inspect.signature(VelocityCommand).parameters) == {
        'sequence_id', 'accepted_at_lower_tick', 'velocity_world'}
    assert ControlSpec().dt == 1/120
