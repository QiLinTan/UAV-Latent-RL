"""Audit must preserve physical action units and separate one-step Q from prefixes."""

import numpy as np

from scripts.audit_full_ft_v1_critic_actor import (
    command_metrics, discounted_return, prefix_bootstrap, ranking_summary,
)


def test_bootstrap_matches_discounted_return_at_each_real_switch():
    rewards = .1 * np.array([3., -2., 4., 250.])
    gamma = .99
    actual = discounted_return(rewards, gamma)
    for cut in (1, 3, len(rewards)):
        tail = discounted_return(rewards[cut:], gamma)
        estimates = prefix_bootstrap(rewards, cut, gamma, [tail, tail, tail])
        np.testing.assert_allclose(estimates, actual)


def test_command_delta_and_away_rate_use_actual_velocity_limits():
    source = np.array([[1., 0., 0.]])
    updated = np.array([[-1., 0., 1.]])
    result = command_metrics(source, updated, np.array([[-3.5, 0., 1.]]), np.array([3.5, 0., 1.]))
    assert np.isclose(result['mean_command_difference_mps'], np.hypot(1.6, .3))
    assert result['mean_planar_heading_difference_degrees'] == 180
    assert result['source_mean_goal_closing_command_mps'] == .8
    assert result['updated_mean_goal_closing_command_mps'] == -.8
    assert result['updated_away_command_fraction'] == 1


def test_long_prefix_ranking_does_not_reuse_initial_action_q():
    case = {'case_id': 'test', 'stage': 0, 'goal_distance_m': 1.}
    def row(kind, value, q, prefix):
        return {'case_id': 'test', 'kind': kind, 'discounted_scaled_return': value,
                'initial_q1_q2_min': [q] * 3, 'prefix_plus_bootstrap_q1_q2_min': [prefix] * 3,
                'outcome': 'timeout', 'trace': [{'reward_scaled': 0.}] * 20,
                'updated_full_15step_estimate': [prefix] * 3}
    rows = [row('source1_then_updated', 3., 4., 0.),
            row('updated_full', -5., 5., -4.),
            row('source_full', 10., 4., 0.),
            row('updated1_then_source', 9., 5., 0.),
            row('source15_then_updated', 2., 4., 1.),
            row('best_q1_1_then_updated', -6., 6., 0.)]
    result = ranking_summary(rows, [case])
    assert result['one_step_common_updated_continuation']['q1']['reversed_order_pairs'] == 1
    # Prefix estimates correctly rank the changed prefix worse, despite higher
    # initial action Q; evaluating both protocols using that initial Q is wrong.
    assert result['fifteen_step_common_updated_continuation']['q1']['reversed_order_pairs'] == 0


def test_full_reward_uses_lower_tick_dt_and_preserves_time_cost():
    from scripts.audit_full_ft_v1_critic_actor import make_full_env
    env, args = make_full_env()
    try:
        env.set_curriculum_stage_override(0)
        observation, _ = env.reset(seed=80)
        assert observation.size == 917
        assert env.base_env._reward_model.integration_dt == 1 / 120
        _, _, terminated, truncated, info = env.step(np.zeros(3))
        assert not (terminated or truncated)
        assert np.isclose(info['reward/time_penalty'], args.time_penalty_weight * 8 / 120)
    finally:
        env.close()
