"""Real Full reset, geometry, gates, successful replay and resume contracts."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest
import torch

from algos.td3.td3_upper_recurrent import TD3UpperRecurrent
from envs.forest.ego_curriculum_env import BANDS, EgoCurriculumFullForestEnv
from envs.forest.obstacle_curriculum import ObstacleCurriculum
from envs.forest.start_curriculum import StartDistanceCurriculum
from envs.map_atlas.forest_pool import ForestPool, COUNTS, digest
from envs.preprocess import preprocess_state
from scripts.audit_stage0_feasibility import make_stage0_env
from scripts.change_td3 import _make_argparser
from scripts.run_change_td3 import build_mode_cmd
from trainers.full_scratch import ScratchAudit, save_training, restore_training, fingerprint
from trainers.td3_trainer import TD3Trainer


def args_for():
    return _make_argparser().parse_args(build_mode_cmd('full_scratch_ego20x10_curr_v1')[3:])


def make_env():
    def factory(**kwargs):
        for key in tuple(kwargs):
            if key.startswith('kinematic_'):
                kwargs.pop(key)
        return EgoCurriculumFullForestEnv(**kwargs)
    args = args_for()
    args.map_id = "ego_forest_20x10_seed43_v1"  # helper requires one catalog entry; factory replaces with frozen pool
    return make_stage0_env(args, env_class=factory)


def agent_for(env):
    return TD3UpperRecurrent(917, 3, 1., base_state_dim=117, history_length=8,
                             history_frame_dim=100, hidden_dim=64, device=torch.device('cpu'))


def test_pool_disjoint_nested_cached_routes_and_sealed_test():
    pool = ForestPool()
    assert {k: len(v) for k, v in pool.manifest['maps'].items()} == {'train': 128, 'validation': 20, 'test': 40}
    hashes = []
    for split, entries in pool.manifest['maps'].items():
        for index, entry in enumerate(entries):
            hashes.append(entry['geometry_sha256'])
            assert entry['seed'] != 43
            for stage, count in enumerate(COUNTS):
                spec = pool.spec(split, index, stage)
                assert len(spec.obstacles) == count and spec.split == split
                path = pool.path_for(split, index, stage)
                np.testing.assert_allclose(path[0], [-8, 0, 1])
                np.testing.assert_allclose(path[-1], [8, 0, 1])
                assert np.all(np.abs(path[:, :2]) < [10, 5])
                xy = np.asarray([o.position[:2] for o in spec.obstacles])
                radii = np.asarray([o.radius for o in spec.obstacles])
                assert (np.linalg.norm(path[:, None, :2] - xy[None], axis=2) - radii[None] - .06).min() >= .2 - 1e-5
                assert spec.obstacles == pool.spec(split, index, 3).obstacles[:count]
    assert len(set(hashes)) == 188
    with pytest.raises(ValueError, match='Sealed test'):
        EgoCurriculumFullForestEnv(pool_split='test', training_course=False)


def test_full_reset_60_combinations_and_actual_active_geometry(tmp_path):
    env = make_env()
    identities = set()
    try:
        # Test-only forced band resets; production retains (.2,.2,.3,.5,1).
        env.start_curriculum.original_probabilities = (0., 0., 0., 0., 1.)
        for seed in (7, 43, 80):
            for stage, count in enumerate(COUNTS):
                for level in range(5):
                    env.start_curriculum.level = level
                    env.obstacle_curriculum.stage = stage
                    state, info = env.reset(seed=seed)
                    identities.add(info['map_id'])
                    assert info['num_obstacles'] == info['active_tree_count'] == len(env.base_env._tree_ids) == count
                    assert info['map_split'] == 'train' and not info['curriculum_mixed_difficulty']
                    if level < 4:
                        assert info['start_curriculum_source'] == 'band'
                        assert BANDS[level][0] <= info['start_curriculum_remaining_path_length'] < BANDS[level][1]
                    else:
                        assert info['start_curriculum_source'] == 'original'
                        np.testing.assert_allclose(env.pos[0], [-8, 0, 1])
                    context = SimpleNamespace(env=env, state=preprocess_state(state), total_steps=0,
                                              args=SimpleNamespace(log_dir=str(tmp_path), scratch_audit_trace_steps=0))
                    audit = ScratchAudit(context)
                    audit.finish()
                    active = env.base_env.MAP_SPEC.obstacles
                    for obstacle in (active[0], env.pool.spec('train', env._pool_index, 3).obstacles[-1]):
                        xy = obstacle.position[:2]
                        hit = p.rayTest([*xy, 5.], [*xy, .1], physicsClientId=env.base_env.CLIENT)[0][0]
                        assert (hit in env.base_env._tree_ids) == (obstacle in active)
                    nearest = min(np.linalg.norm(env.pos[0, :2]-o.position[:2])-o.radius-env.base_env.COLLISION_R for o in active)
                    assert env.base_env._computeNearestTreeClearance(env.pos[0]) == pytest.approx(nearest, abs=1e-5)
                    for _ in range(3):
                        state, reward, term, trunc, step_info = env.step(np.array([[.4, -.2, .1]], dtype=np.float32))
                        assert np.isfinite(state).all() and np.isfinite(reward)
                        assert step_info['lower_steps_executed'] == 8
                    # Next iteration resets after nonzero PID/reference/physical history.
        assert len(identities) == 3
        active = env.base_env.MAP_SPEC.obstacles[0]
        assert env._place_kinematic_state(np.array([*active.position[:2], 1.]), np.zeros(3))
        env.obstacle_curriculum.stage = 0
        env.reset(seed=80)
        inactive = env.pool.spec('train', env._pool_index, 3).obstacles[-1]
        assert not env._place_kinematic_state(np.array([*inactive.position[:2], 1.]), np.zeros(3))
        vec = env.base_env._getDroneStateVector(0).copy()
        vec[:3] = [0, 5.01, 1.]
        assert env.base_env._classifyDoneReason(vec, collision=False) == 'xy_bound'
    finally:
        env.close()


def test_gates_exclude_original_mix_short_starts_stale_density_and_easy_mix():
    start = StartDistanceCurriculum(window=2, minimum_episodes=2, bands=BANDS)
    for source in ('original', 'planner_fallback'):
        for _ in range(4):
            start.record(source=source, success=True, collision=False)
    assert start.level == 0 and start.level_episodes == 0
    for _ in range(4):
        for _ in range(2):
            start.record(source='band', success=True, collision=False)
    assert start.complete and start.level == 4
    density = ObstacleCurriculum(route_counts=COUNTS, window=2, minimum_episodes=2)
    def outcome(stage=0, complete=True, primary=True):
        return density.record(episode_stage=stage, start_level=0, start_complete=complete,
                              primary_difficulty=primary, success=True, collision=False)
    assert not outcome(complete=False) and not outcome(primary=False) and not outcome(stage=3)
    assert density.stage_episodes == 0
    assert not outcome() and outcome()
    assert density.stage == 1 and density.stage_episodes == 0
    assert not outcome(stage=0) and density.stage_episodes == 0


def test_real_full_success_enters_density_replay_failure_excluded():
    env = make_env()
    args = args_for()
    args.buffer_size = 2048
    args.success_replay_size = 512
    args.start_timesteps = 0
    args.expl_noise_start = args.expl_noise_end = 0.
    agent = agent_for(env)
    # Isolated test controller. Never used in production run or saved learners.
    def controller(_):
        delta = env.TARGET_POS-env.pos[0]
        cmd = 1.5 * delta - .6 * env.vel[0]
        action = cmd / np.array([.8, .8, .3])
        return np.clip(action, -1., 1.).astype(np.float32)
    agent.select_action = controller
    try:
        trainer = TD3Trainer(env, agent, args)
        for step in range(1, 151):
            trainer.total_steps = step
            trainer.step_env()
            if trainer.last_info.get('done_reason') == 'success':
                break
        else:
            pytest.fail('Deterministic near-goal Full probe did not succeed')
        size = trainer.buffer.size
        assert trainer.success_buffer.size == size
        assert trainer.success_buffer.buffers[10].size == size
        assert all(trainer.success_buffer.buffers[k].size == 0 for k in (20, 30, 40))
        assert trainer.success_buffer.buffers[10].not_done[size-1, 0] == 0
        # A real height-bound failure after success cannot add any success samples.
        agent.select_action = lambda _: np.array([0., 0., -1.], dtype=np.float32)
        for step in range(size+1, size+601):
            trainer.total_steps = step
            trainer.step_env()
            if trainer.last_info.get('done_reason') == 'height_bound':
                break
        else:
            pytest.fail('Failure replay probe did not terminate')
        assert trainer.success_buffer.size == size
        # Original-start hover produces the actual 40s timeout rule and mask.
        env.start_curriculum.level = 4
        observation, reset_info = env.reset()
        trainer.state = preprocess_state(observation)
        trainer._episode_success_transitions.clear()
        trainer._reset_episode_coverage(reset_info)
        trainer.episode_step = 0
        trainer.episode_return = 0.
        agent.select_action = lambda _: np.zeros(3, dtype=np.float32)
        for tick in range(1, 610):
            trainer.total_steps += 1
            trainer.step_env()
            if trainer.last_info.get('done_reason') == 'timeout':
                break
        else:
            pytest.fail('40s hover timeout did not terminate')
        assert tick in (600, 601)
        assert trainer.success_buffer.size == size
        assert trainer.buffer.not_done[(trainer.buffer.ptr-1) % trainer.buffer.max_size, 0] == 0
        assert agent.total_it == 0
    finally:
        env.close()


def test_pool_resume_preserves_nonempty_adam_lagged_targets_and_next_update(tmp_path):
    env = make_env()
    other_env = make_env()
    args = args_for()
    args.ckpt_dir = str(tmp_path)
    args.buffer_size = 64
    args.success_replay_size = 64
    args.total_steps = 2010
    def context(environment):
        trainer = TD3Trainer(environment, agent_for(environment), copy.deepcopy(args))
        rng = np.random.default_rng(6)
        for _ in range(64):
            trainer.buffer.push(rng.normal(size=917), rng.uniform(-.5, .5, 3), 1., rng.normal(size=917), False)
        trainer.total_steps = 2000
        return trainer
    try:
        first = context(env)
        for _ in range(4):
            first.agent.train(first.buffer, batch_size=8)
        env.start_curriculum.level = 2
        env.start_curriculum.record(source='band', success=True, collision=False)
        env.obstacle_curriculum.stage = 1
        expected_sampler = env.sampler_state()
        expected_rng = np.random.default_rng()
        expected_rng.bit_generator.state = expected_sampler['rng']
        expected_next = int(expected_rng.integers(128))
        save_training(first)
        first.agent.train(first.buffer, batch_size=8)
        second = context(other_env)
        second.args.resume_training = str(tmp_path / 'training_state.pt')
        restore_training(second)
        assert second.env._pool_index == expected_next
        assert second.env._pool_draw_count == expected_sampler['draws'] + 1
        assert second.env.start_curriculum.level == 2 and list(second.env.start_curriculum.successes) == [True]
        assert second.env.obstacle_curriculum.stage == 1
        assert second.agent.total_it == 4 and len(second.agent.actor_optimizer.state) > 0
        assert fingerprint(second.agent.critic) != fingerprint(second.agent.critic_target)
        np.testing.assert_array_equal(first.buffer.state, second.buffer.state)
        second.agent.train(second.buffer, batch_size=8)
        for name in ('actor', 'critic', 'actor_target', 'critic_target'):
            assert fingerprint(getattr(first.agent, name)) == fingerprint(getattr(second.agent, name))
    finally:
        env.close()
        other_env.close()


def test_new_preset_preserves_algorithm_rewards_and_warmup():
    base = vars(_make_argparser().parse_args(build_mode_cmd('full_scratch_curr_v1')[3:]))
    new = vars(args_for())
    allowed = {'full_scratch_curr_v1', 'full_scratch_ego20x10_curr_v1', 'curriculum', 'route_count_curriculum', 'map_id', 'eval_map_id',
               'eval_episodes', 'success_replay_difficulties', 'log_dir', 'ckpt_dir', 'trajectory_output_dir',
               'collision_diagnostic_output_dir', 'timeout_diagnostic_output_dir', 'eval_terminal_coverage_output_dir'}
    assert {key for key in base if base[key] != new[key]} <= allowed
    assert new['start_timesteps'] == new['update_after'] == 5000
    assert new['buffer_size'] == 60000 and new['success_replay_size'] == 20000
    assert new['success_replay_difficulties'] == [10, 20, 30, 40]
    assert new['eval_episodes'] == 20 and new['eval_interval'] == 5000
    assert not new['load_checkpoint'] and not new['full_ft_v1']


def test_last_short_episode_cannot_certify_density_and_evaluation_cannot_advance():
    env = EgoCurriculumFullForestEnv.__new__(EgoCurriculumFullForestEnv)
    env.training_course = True
    env.start_curriculum = StartDistanceCurriculum(window=1, minimum_episodes=1, bands=BANDS)
    env.start_curriculum.level = 3
    env.obstacle_curriculum = ObstacleCurriculum(route_counts=COUNTS, window=1, minimum_episodes=1)
    env._start_level = 3
    env._start_source = 'band'
    env._density_episode_stage = 0
    assert not env.record_episode_outcome(True)
    assert env.start_curriculum.complete and env.obstacle_curriculum.stage_episodes == 0
    env._start_level = 4
    env._start_source = 'original'
    assert env.record_episode_outcome(True) and env.obstacle_curriculum.stage == 1
    env._density_episode_stage = 1
    env.training_course = False
    assert not env.record_episode_outcome(True) and env.obstacle_curriculum.stage_episodes == 0


def test_monitoring_uses_world_distance_instead_of_clipped_goal_features():
    from trainers.full_scratch import world_goal_delta
    states = np.zeros((2, 917), dtype=np.float32)
    states[:, 0] = [-16., 15.]
    states[:, 12] = [1., .05]  # environmental goal features clipped/scaled
    np.testing.assert_allclose(np.linalg.norm(world_goal_delta(states, [8, 0, 1]), axis=1), [16., .5])
