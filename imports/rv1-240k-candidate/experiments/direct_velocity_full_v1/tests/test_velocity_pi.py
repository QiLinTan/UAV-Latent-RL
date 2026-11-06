import numpy as np
import pytest

from experiments.direct_velocity_full_v1.contracts import ModelParameters, ControlSpec
from experiments.direct_velocity_full_v1.controls.velocity_tracking import VelocityPI, constrain_force


def test_hover_gravity_without_position():
    pi = VelocityPI(ModelParameters())
    force, diagnostics, candidate = pi.propose([0, 0, 0], [0, 0, 0])
    np.testing.assert_allclose(force, [0, 0, .2646])
    np.testing.assert_array_equal(candidate, [0, 0, 0])
    assert set(diagnostics).isdisjoint({'target_pos', 'p_ref', 'reference_position'})


def test_force_cone_and_total_limit_random_vectors():
    model, spec = ModelParameters(), ControlSpec()
    for raw in np.random.default_rng(12).normal(size=(2000, 3)):
        force, _ = constrain_force(raw, model, spec)
        assert np.isfinite(force).all() and force[2] > 0
        assert np.linalg.norm(force) <= model.max_force_n * spec.max_force_fraction + 1e-12
        assert np.arctan2(np.linalg.norm(force[:2]), force[2]) <= spec.max_tilt_rad + 1e-12


def test_integral_and_antiwindup_no_saturation_bias():
    pi = VelocityPI(ModelParameters())
    for _ in range(2000):
        _, diagnostic, candidate = pi.propose([.8, 0, .3], [-10, 0, -10])
        pi.commit(candidate)
    assert diagnostic['force_limited']
    np.testing.assert_array_equal(pi.integral, [0, 0, 0])
    pi.integral[:] = [.5, .5, .5]
    _, _, candidate = pi.propose([.8, .8, .3], [-10, -10, -10])
    assert np.all(abs(candidate) <= abs(pi.integral))


def test_integral_bounds_allocation_feedback_and_reset():
    pi = VelocityPI(ModelParameters())
    for _ in range(4000):
        _, _, candidate = pi.propose([.1, .1, .1], [0, 0, 0])
        pi.commit(candidate)
    assert np.all(abs(pi.integral) <= np.asarray(pi.spec.integral_limit_mps2))
    before = pi.integral.copy()
    _, _, candidate = pi.propose([.1, .1, .1], [0, 0, 0], allocation_saturated=True)
    np.testing.assert_array_equal(before, candidate)
    pi.reset()
    np.testing.assert_array_equal(pi.integral, [0, 0, 0])


def test_invalid_force_inputs_leave_integral_unchanged():
    pi = VelocityPI(ModelParameters())
    for bad in ([np.nan, 0, 0], [np.inf, 0, 0], [[0, 0, 0]]):
        before = pi.integral.copy()
        with pytest.raises(ValueError):
            pi.propose(bad, [0, 0, 0])
        np.testing.assert_array_equal(pi.integral, before)


def test_U04_same_motion_state_same_force_no_position_dependency():
    left, right = VelocityPI(ModelParameters()), VelocityPI(ModelParameters())
    for _ in range(100):
        a, _, ia = left.propose([.5, -.2, .1], [.1, -.1, .01])
        b, _, ib = right.propose([.5, -.2, .1], [.1, -.1, .01])
        np.testing.assert_array_equal(a, b)
        left.commit(ia)
        right.commit(ib)
