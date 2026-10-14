import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from algos.td3.td3_upper_recurrent import TD3UpperRecurrent
from data.replay_buffer import ReplayBuffer
from envs.forest.start_curriculum import StartDistanceCurriculum
from envs.forest.obstacle_curriculum import ObstacleCurriculum
from envs.forest.start_curriculum_env import StartObstacleCurriculumFullForestEnv
from envs.learned_upper_dslpid import LearnedUpperDSLPIDEnv, KinematicUpperEnv
from scripts.change_td3 import _make_argparser
from scripts.run_change_td3 import build_mode_cmd
from trainers.full_scratch import save_training, restore_training, fingerprint, validate_scratch_args


def test_preset_inherits_learning_contract():
    base = vars(_make_argparser().parse_args(build_mode_cmd(
        'upper_dslpid_recurrent_multitree_terminal_v2_noroute_startobscurr_v1_h8_k1')[3:]))
    full = vars(_make_argparser().parse_args(build_mode_cmd('full_scratch_curr_v1')[3:]))
    allowed = {'upper_execution_model', 'full_scratch_curr_v1', 'curriculum_simple_mix_probability',
               'eval_interval', 'ckpt_interval', 'log_dir', 'ckpt_dir', 'trajectory_output_dir',
               'collision_diagnostic_output_dir', 'timeout_diagnostic_output_dir',
               'eval_terminal_coverage_output_dir'}
    assert {k for k in base if base[k] != full[k]} <= allowed
    assert full['start_timesteps'] == full['update_after'] == 5000
    assert not full['full_ft_v1'] and not full['load_checkpoint']
    assert StartObstacleCurriculumFullForestEnv.step.__qualname__ == 'StartObstacleCurriculumMixin.step'
    assert KinematicUpperEnv not in StartObstacleCurriculumFullForestEnv.__mro__
    assert LearnedUpperDSLPIDEnv in StartObstacleCurriculumFullForestEnv.__mro__


def test_overwrite_and_pretrained_rejected(tmp_path):
    args = _make_argparser().parse_args(build_mode_cmd('full_scratch_curr_v1')[3:])
    args.ckpt_dir = str(tmp_path / 'full_scratch_curr_v1_test')
    validate_scratch_args(args)
    from pathlib import Path
    Path(args.ckpt_dir).mkdir()
    (Path(args.ckpt_dir) / 'model_initial_actor').write_text('existing')
    with pytest.raises(FileExistsError):
        validate_scratch_args(args)
    args.load_checkpoint = 'K1'
    with pytest.raises(ValueError):
        validate_scratch_args(args)


class FakeEnv:
    def __init__(self):
        self.start_curriculum = StartDistanceCurriculum()
        self.obstacle_curriculum = ObstacleCurriculum()
        self._start_rng = np.random.default_rng(4)
        self.base_env = SimpleNamespace(_rng=np.random.default_rng(5), _reset_count=1)

    def reset(self):
        return np.zeros(12, dtype=np.float32), {}


def test_resume_nonempty_adam_targets_replay_curricula_and_next_update(tmp_path):
    def trainer():
        agent = TD3UpperRecurrent(11, 3, 1., base_state_dim=5, history_length=2,
                                 history_frame_dim=3, hidden_dim=8, device=torch.device('cpu'))
        replay = ReplayBuffer(11, 3, max_size=32)
        rng = np.random.default_rng(6)
        for _ in range(32):
            replay.push(rng.normal(size=11), rng.uniform(-.5, .5, 3), 1., rng.normal(size=11), False)
        return SimpleNamespace(agent=agent, buffer=replay, success_buffer=None,
            env=FakeEnv(), callbacks=[], total_steps=2000, episode_step=3,
            args=SimpleNamespace(ckpt_dir=str(tmp_path), total_steps=2010, resume_training=None),
            _reset_episode_coverage=lambda info: None, _episode_success_transitions=[])
    first = trainer()
    for _ in range(4):
        first.agent.train(first.buffer, batch_size=8)
    first.env.start_curriculum.level = 2
    first.env.start_curriculum.record(source='band', success=True, collision=False)
    first.env.obstacle_curriculum.stage = 1
    save_training(first)
    first.agent.train(first.buffer, batch_size=8)
    second = trainer()
    second.args.resume_training = str(tmp_path / 'training_state.pt')
    restore_training(second)
    assert second.total_steps == 2000 and second.agent.total_it == 4
    assert len(second.agent.actor_optimizer.state) > 0
    assert second.env.start_curriculum.level == 2
    assert list(second.env.start_curriculum.successes) == [True]
    assert second.env.obstacle_curriculum.stage == 1
    assert fingerprint(second.agent.critic) != fingerprint(second.agent.critic_target)
    second.agent.train(second.buffer, batch_size=8)
    assert fingerprint(first.agent.critic) == fingerprint(second.agent.critic)
    assert fingerprint(first.agent.actor) == fingerprint(second.agent.actor)
    assert fingerprint(first.agent.critic_target) == fingerprint(second.agent.critic_target)


def test_full_reset_clears_pid_packet_history_and_all_obstacle_stages(tmp_path):
    from scripts.audit_stage0_feasibility import make_stage0_env
    from scripts.diagnose_terminal_reward_rollouts import preset_args
    from envs.preprocess import preprocess_state
    from trainers.full_scratch import ScratchAudit

    def full_factory(**kwargs):
        for key in tuple(kwargs):
            if key.startswith('kinematic_'):
                kwargs.pop(key)
        return StartObstacleCurriculumFullForestEnv(**kwargs)

    env = make_stage0_env(preset_args('full_scratch_curr_v1'), env_class=full_factory)
    try:
        env.reset(seed=43)
        for _ in range(10):
            _, _, _, _, info = env.step(np.array([[.5, -.3, .2]], dtype=np.float32))
        assert info['lower_steps_executed'] == 8 and env.last_reference_packet is not None
        assert env.teacher.control_counter > 0
        state, _ = env.reset(seed=43)
        trainer = SimpleNamespace(env=env, state=preprocess_state(state), total_steps=0,
                                  args=SimpleNamespace(log_dir=str(tmp_path), scratch_audit_trace_steps=0))
        ScratchAudit(trainer)
        env.start_curriculum.level = 4
        for stage, count in enumerate((1, 2, 3, 5)):
            env.obstacle_curriculum.stage = stage
            for seed in (0, 43, 80):
                _, info = env.reset(seed=seed)
                assert info['start_curriculum_source'] == 'original'
                assert info['start_curriculum_initial_distance'] == 7.
                assert info['route_blocking_tree_count'] == count
                assert not info['curriculum_mixed_difficulty']
    finally:
        env.close()
