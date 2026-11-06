"""Verify diagnostic reconstruction and geometry, without training."""
from types import SimpleNamespace

import numpy as np

from scripts.audit_full_ego20x10_task_contract import COSTS, trace_reward, smooth_route, segment_clear


def test_saved_terminal_outcome_overrides_rounded_lower_time(tmp_path):
    class Model:
        def compute(self, **kwargs):
            terminal = kwargs['terminal_reason'] != 'running'
            value = -180. if terminal else .01
            return value, {'progress_reward': 0. if terminal else .01,
                           'terminal_outcome_reward': -180. if terminal else 0.,
                           **{key: 0. for key in COSTS}}, 16.
    env = SimpleNamespace(base_env=SimpleNamespace(_reward_model=Model()),
                          TARGET_POS=np.array([8, 0, 1]), START_POS=np.array([-8, 0, 1]),
                          GOAL_TOLERANCE=.2, SAFE_DISTANCE=.35)
    path = tmp_path/'trace.npz'
    np.savez(path, lower_reward=[.01, -180.], lower_position=[[-8, 0, 1]]*2,
             lower_velocity=np.zeros((2, 3)), lower_rpy=np.zeros((2, 3)),
             lower_collision=[False]*2, lower_time_seconds=[40.000000000001, 40.01],
             lower_clearance=[np.inf]*2, upper_reward=[-179.99])
    record = trace_reward(path, {'initial_position': [-8, 0, 1], 'done_reason': 'timeout'}, env)
    assert record['maximum_lower_reward_error'] == 0
    assert record['reward_components']['terminal_outcome_reward'] == -180.


def test_smoothing_preserves_clearance_around_blocking_tree():
    obstacle = SimpleNamespace(position=[0, 0, 1], radius=.25)
    path = np.array([[-2, 0, 1], [-1, 1, 1], [0, 1, 1], [1, 1, 1], [2, 0, 1]], dtype=float)
    assert not segment_clear(path[0], path[-1], [obstacle])
    shortened = smooth_route(path, [obstacle])
    assert len(shortened) >= 3
    assert all(segment_clear(a, b, [obstacle]) for a, b in zip(shortened, shortened[1:]))
    np.testing.assert_array_equal(shortened[[0, -1]], path[[0, -1]])
    np.testing.assert_array_equal(smooth_route(path, []), path[[0, -1]])
