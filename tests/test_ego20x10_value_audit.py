"""Audit harness invariants; no production learners or sealed test maps."""
import numpy as np
import pybullet as p

from scripts.audit_ego20x10_value import (
    LOGS, load, env_for, reconstruct_state, rows, python_state, restore_physics,
)
from envs.preprocess import preprocess_state


def test_diagnostic_full_reward_matches_actual_first_replay_transition():
    saved = load(6000)
    replay = saved['replay']['arrays']
    env = env_for(saved, training_course=False, pool_split='train')
    try:
        assert env.base_env._reward_model.integration_dt == 1/120
        observation, _, error = reconstruct_state(env, 0, replay, rows(LOGS/'p0_audit/resets.jsonl'))
        assert error == 0
        observation, reward, terminated, truncated, _ = env.step(replay['action'][0].reshape(1,3))
        np.testing.assert_array_equal(preprocess_state(observation), replay['next_state'][0])
        np.testing.assert_allclose(reward*.1, replay['reward'][0,0], rtol=1e-6)
        assert replay['not_done'][0,0] == 1-float(terminated or truncated)
    finally:
        env.close()


def test_fresh_action_reconstruction_preserves_physics_pid_packet_and_history():
    saved = load(6000)
    replay = saved['replay']['arrays']
    env = env_for(saved, training_course=False, pool_split='train')
    try:
        observation, _, error = reconstruct_state(env, 10, replay, rows(LOGS/'p0_audit/resets.jsonl'))
        assert error == 0 and env.teacher.control_counter > 0
        assert env.last_reference_packet is not None
        action = np.array([[.3,-.2,.4]],dtype=np.float32)
        first = env.step(action)
        rebuilt, _, error = reconstruct_state(env, 10, replay, rows(LOGS/'p0_audit/resets.jsonl'))
        assert error == 0
        np.testing.assert_array_equal(rebuilt, observation)
        np.testing.assert_array_equal(env._upper_observation(env.base_env._computeObs()), observation)
        second = env.step(action)
        np.testing.assert_array_equal(first[0], second[0])
        assert first[1:4] == second[1:4]
    finally:
        env.close()
