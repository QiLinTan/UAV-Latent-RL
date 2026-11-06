"""Full scratch contract, read-only P0 tracing, and resumable training state.

Snapshots are trusted local torch files. Resumption starts a new physical
episode; it preserves learners, replay, schedules and completed-episode gates.
"""
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from envs.preprocess import preprocess_state
from trainers.paired_trajectory_trace import PairedTrajectoryTrace


def fingerprint(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def world_goal_delta(states, target_position):
    # Invert only the KIN position preprocessing; goal features are normalized
    # and clipped by the environment and cannot be used as metres.
    positions = states[:, :3] * .5 + np.array([0., 0., 1.])
    return np.asarray(target_position) - positions


def policy_diagnostics(agent, states, actions, target_position=None):
    """Deterministic fixed-state probe; never consumes exploration/replay RNG."""
    with torch.no_grad():
        states_t = torch.as_tensor(states, device=agent.device)
        policy = agent.actor(states_t).cpu().numpy()
        q1, q2 = agent.critic(states_t, torch.as_tensor(actions, device=agent.device))
    radial = states[:, 12:14] if target_position is None else world_goal_delta(states, target_position)[:, :2]
    radial = radial / np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
    metrics = {
        'actor_planar_boundary_fraction': float(np.mean(np.linalg.norm(policy[:, :2], axis=1) >= .999)),
        'actor_axis_saturation_fraction': float(np.mean(np.abs(policy) >= .999)),
        'actor_radial_planar_command_mps': float(np.mean(np.sum(policy[:, :2] * radial, axis=1)) * .8),
        'actor_vertical_command_mps': float(np.mean(policy[:, 2]) * .3),
        'actual_radial_planar_velocity_mps': float(np.mean(np.sum(states[:, 6:8] * 2 * radial, axis=1))),
        'actual_vertical_velocity_mps': float(np.mean(states[:, 8]) * 2),
    }
    for axis, label in enumerate(('x', 'y', 'z')):
        metrics[f'actor_{label}_saturation_fraction'] = float(np.mean(np.abs(policy[:, axis]) >= .999))
    for key, value in (('q1', q1), ('q2', q2)):
        metrics[f'{key}_mean'] = float(value.mean().item())
        metrics[f'{key}_std'] = float(value.std(unbiased=False).item())
        for label, number in zip(('min', 'p05', 'p50', 'p95', 'max'),
                                 np.quantile(value.cpu().numpy(), [0, .05, .5, .95, 1])):
            metrics[f'{key}_{label}'] = float(number)
    return metrics, policy


def validate_scratch_args(args):
    required = (args.learned_upper_dslpid and args.upper_recurrent
                and args.upper_execution_model == 'full'
                and args.start_distance_curriculum and args.obstacle_curriculum_after_start)
    if not required or args.full_ft_v1 or args.stage01_finetune:
        raise ValueError('Full scratch requires Full recurrent start/obstacle curriculum')
    if args.load_checkpoint or args.full_ft_actor_source:
        raise ValueError('Full scratch forbids pretrained weights; use --resume_training for continuation')
    if args.curriculum_simple_mix_probability != 0 or args.eval_interval != 5000:
        raise ValueError('Full scratch requires no hidden simple mix and 5k evaluation')
    if getattr(args, 'full_scratch_ego20x10_curr_v1', False):
        if args.full_scratch_curr_v1 or args.eval_episodes != 20 or tuple(args.success_replay_difficulties) != (10, 20, 30, 40):
            raise ValueError('New pool protocol requires exclusive preset, 20 validation maps and density replay strata')
    if args.evaluate_only:
        raise ValueError('Use ordinary Full evaluation mode for saved scratch policies')
    root = Path(args.ckpt_dir).resolve()
    expected = 'full_scratch_ego20x10_curr_v1' if getattr(args, 'full_scratch_ego20x10_curr_v1', False) else 'full_scratch_curr_v1'
    if not root.name.startswith(expected):
        raise ValueError(f'Scratch checkpoints require a dedicated {expected} directory')
    if not args.resume_training and root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing scratch checkpoints: {root}')
    if args.resume_training and Path(args.resume_training).resolve().parent != root:
        raise ValueError('Resume from this run directory; do not mix experiment outputs')


def buffer_state(buffer):
    if buffer is None:
        return None
    if hasattr(buffer, 'buffers'):
        return {'strata': {k: buffer_state(v) for k, v in buffer.buffers.items()},
                'last_sample_counts': buffer.last_sample_counts}
    return {'size': buffer.size, 'ptr': buffer.ptr, 'max_size': buffer.max_size,
            'arrays': {k: getattr(buffer, k)[:buffer.size].copy()
                       for k in ('state', 'action', 'next_state', 'reward', 'not_done')}}


def restore_buffer(buffer, state):
    if state is None:
        assert buffer is None
        return
    if 'strata' in state:
        for k, value in state['strata'].items():
            restore_buffer(buffer.buffers[k], value)
        buffer.last_sample_counts = state['last_sample_counts']
    else:
        if buffer.max_size != state['max_size']:
            raise ValueError('Replay capacity changed across resume')
        buffer.size, buffer.ptr = state['size'], state['ptr']
        for k, value in state['arrays'].items():
            getattr(buffer, k)[:buffer.size] = value


def protocol(args):
    mutable = {'total_steps', 'resume_training', 'scratch_audit_trace_steps',
               'log_dir', 'ckpt_dir', 'trajectory_output_dir',
               'collision_diagnostic_output_dir', 'timeout_diagnostic_output_dir',
               'eval_terminal_coverage_output_dir'}
    if not getattr(args, 'full_scratch_ego20x10_curr_v1', False):
        mutable |= {'full_scratch_ego20x10_curr_v1', 'forest_pool_path'}
    return {k: v for k, v in vars(args).items() if k not in mutable}


def save_training(trainer):
    env = trainer.env
    callback_states = {i: copy.deepcopy(cb.__dict__) for i, cb in enumerate(trainer.callbacks)
                       if type(cb).__name__ in ('BestCheckpointCallback', 'ConvergenceGuardCallback')}
    # Live callbacks contain references to other callbacks; preserve only state.
    for value in callback_states.values():
        value.pop('final_eval_callback', None)
        value.pop('best_callback', None)
        value.pop('save_dir', None)
    payload = dict(schema=1, protocol=protocol(trainer.args),
                   agent=trainer.agent.__dict__, steps=trainer.total_steps,
                   replay=buffer_state(trainer.buffer), success=buffer_state(trainer.success_buffer),
                   start=env.start_curriculum.__dict__, obstacle=env.obstacle_curriculum.__dict__,
                   start_rng=env._start_rng.bit_generator.state,
                   map_rng=env.base_env._rng.bit_generator.state,
                   reset_count=env.base_env._reset_count,
                   numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                   torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                   callbacks=callback_states,
                   interrupted_episode_steps=trainer.episode_step,
                   resume_semantics='new_episode_preserve_training_state')
    if hasattr(env, 'sampler_state'):
        payload['pool_sampler'] = env.sampler_state()
    path = Path(trainer.args.ckpt_dir) / 'training_state.pt'
    torch.save(payload, str(path) + '.tmp')
    Path(str(path) + '.tmp').replace(path)
    if hasattr(env, 'sampler_state'):
        shutil.copyfile(path, path.with_name(f'training_state_step{trainer.total_steps}.pt'))


def restore_training(trainer):
    saved = torch.load(trainer.args.resume_training, map_location=trainer.agent.device, weights_only=False)
    if saved['schema'] != 1 or saved['protocol'] != protocol(trainer.args):
        raise ValueError('Resume protocol differs from saved experiment')
    if trainer.args.total_steps <= saved['steps']:
        raise ValueError('--total_steps must exceed saved cumulative upper transitions')
    trainer.agent.__dict__.update(saved['agent'])
    restore_buffer(trainer.buffer, saved['replay'])
    restore_buffer(trainer.success_buffer, saved['success'])
    env = trainer.env
    env.start_curriculum.__dict__.update(saved['start'])
    env.obstacle_curriculum.__dict__.update(saved['obstacle'])
    env._start_rng.bit_generator.state = saved['start_rng']
    env.base_env._rng.bit_generator.state = saved['map_rng']
    env.base_env._reset_count = saved['reset_count']
    for index, state in saved['callbacks'].items():
        state = {k: v for k, v in state.items() if k != "save_dir"}
        trainer.callbacks[index].__dict__.update(state)
    if hasattr(env, 'restore_sampler_state'):
        if 'pool_sampler' not in saved:
            raise ValueError('New forest run requires pool sampler snapshot')
        env.restore_sampler_state(saved['pool_sampler'])
    trainer.total_steps = saved['steps']
    resume_evidence = dict(saved_step=saved['steps'], actor=fingerprint(trainer.agent.actor),
                           critic=fingerprint(trainer.agent.critic), total_it=trainer.agent.total_it,
                           replay_size=trainer.buffer.size, replay_ptr=trainer.buffer.ptr,
                           start_level=env.start_curriculum.level, density_stage=env.obstacle_curriculum.stage,
                           success_sizes={str(k): b.size for k, b in trainer.success_buffer.buffers.items()}
                           if hasattr(trainer.success_buffer, 'buffers') else None,
                           pool_sampler=env.sampler_state() if hasattr(env, 'sampler_state') else None,
                           semantics='new_physical_episode; interrupted episode discarded from success replay/gates')
    (Path(trainer.args.ckpt_dir) / f'resume_at_{saved["steps"]}.json').write_text(json.dumps(resume_evidence, indent=2) + '\n')
    trainer.state, reset_info = env.reset()
    trainer.state = preprocess_state(trainer.state.reshape(-1))
    trainer._reset_episode_coverage(reset_info)
    trainer._episode_success_transitions.clear()
    trainer.episode_step = 0
    trainer.episode_return = 0
    np.random.set_state(saved['numpy_rng'])
    random.setstate(saved['python_rng'])
    torch.set_rng_state(saved['torch_rng'].cpu())
    if saved['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    print(f"[Resume] cumulative_step={trainer.total_steps}; preserved networks/targets/optimizers/replay/curricula; new physical episode")


def prepare_scratch_training(trainer):
    root = Path(trainer.args.ckpt_dir)
    root.mkdir(parents=True, exist_ok=True)
    if trainer.args.resume_training:
        restore_training(trainer)
    else:
        trainer.agent.save(str(root / 'model_initial'))
        initial = dict(actor=fingerprint(trainer.agent.actor), critic=fingerprint(trainer.agent.critic),
                       actor_optimizer_states=len(trainer.agent.actor_optimizer.state),
                       critic_optimizer_states=len(trainer.agent.critic_optimizer.state),
                       total_it=trainer.agent.total_it, pretrained_source=None,
                       args=vars(trainer.args))
        (root / 'initialization.json').write_text(json.dumps(initial, indent=2) + '\n')
    trainer.scratch_audit = ScratchAudit(trainer)


class ScratchAudit:
    def __init__(self, trainer):
        self.trainer = trainer
        self.root = Path(trainer.args.log_dir) / 'p0_audit'
        self.root.mkdir(parents=True, exist_ok=True)
        self.trace = None
        self.rows = []
        self.probe_path = self.root / 'fixed_policy_probe.npz'
        self.pool_mode = hasattr(trainer.env, 'pool')
        self.saved_reasons = set()
        for path in (self.root / 'traces').glob('*.json'):
            self.saved_reasons.add(json.loads(path.read_text()).get('done_reason'))
        self.on_reset()

    def on_reset(self):
        env = self.trainer.env
        assert env.observation_space.shape == (1, 917)
        assert env.CTRL_FREQ == 15 and env.lower_ctrl_freq == 120
        assert env.base_env.CURRICULUM_SIMPLE_MIX_PROBABILITY == 0
        for value in (env.vel, env.ang_v, env.rpy, env.reference_velocity,
                      env.reference_acceleration, env.command_velocity, env.last_motor_action):
            np.testing.assert_allclose(value, 0, atol=1e-7)
        np.testing.assert_allclose(env.reference_position, env.pos[0], atol=1e-7)
        assert env.runtime_lower_step == 0 and env.packet_generation_count == 0
        assert env.last_reference_packet is None and env.last_reference_sample is None
        assert not env.base_env._checkCollision()
        fresh = type(env.teacher)(drone_model=env.base_env.DRONE_MODEL)
        for name, value in vars(fresh).items():
            if isinstance(value, (np.ndarray, int, float)):
                np.testing.assert_allclose(getattr(env.teacher, name), value)
        history = np.asarray(env._semantic_history)
        np.testing.assert_allclose(history, np.repeat(history[:1], len(history), axis=0))
        np.testing.assert_allclose(history[-1], env._semantic_frame(env.base_env._computeObs()))
        np.testing.assert_allclose(self.trainer.state, preprocess_state(env._upper_observation(env.base_env._computeObs()).reshape(-1)))
        row = dict(step=self.trainer.total_steps, start=env.pos[0].tolist(),
                   source=env._start_source, level=env.start_curriculum.level,
                   stage=env.obstacle_curriculum.stage, clearance=env.base_env._last_clearance,
                   distance=env._start_info()['start_curriculum_initial_distance'], reset_pass=True)
        if self.pool_mode:
            row.update(env._pool_info())
        with (self.root / 'resets.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        if self.pool_mode or self.trainer.total_steps < self.trainer.args.scratch_audit_trace_steps:
            self.trace = PairedTrajectoryTrace(env, None, trace_id=f"episode_at_{self.trainer.total_steps}")
            self.trace.install()
        self.rows = []

    def transition(self, pre_state, action, next_state, reward, terminated, truncated, info):
        trainer, env = self.trainer, self.trainer.env
        index = (trainer.buffer.ptr - 1) % trainer.buffer.max_size
        assert np.isfinite(next_state).all() and np.isfinite(reward)
        assert next_state.shape == (917,)
        assert info['reference_packet_valid']
        ticks = info['lower_steps_executed']
        assert 1 <= ticks <= 8 and (ticks == 8 or terminated or truncated)
        assert np.linalg.norm(action[:2]) <= 1.000001
        np.testing.assert_allclose(trainer.buffer.action[index], action)
        np.testing.assert_allclose(trainer.buffer.state[index], pre_state)
        np.testing.assert_allclose(trainer.buffer.next_state[index], next_state)
        np.testing.assert_allclose(trainer.buffer.reward[index, 0], reward * trainer.args.reward_scale, rtol=1e-6)
        assert trainer.buffer.not_done[index, 0] == 1 - float(terminated or truncated)
        history = next_state[env.base_upper_obs_dim:].reshape(env.semantic_history_length, env.semantic_frame_dim)
        old_history = pre_state[env.base_upper_obs_dim:].reshape(history.shape)
        np.testing.assert_allclose(history[:-1], old_history[1:])
        np.testing.assert_allclose(history[-1], env._semantic_frame(
            env.base_env._computeObs(), info['reference_tracking_error_mean']))
        if self.trace is not None:
            policy = trainer.agent.select_action(pre_state)
            self.trace.record_upper(action, policy, reward, info, pre_observation=pre_state)
            np.testing.assert_allclose(sum(r['reward'] for r in self.trace.lower[-ticks:]), reward, rtol=1e-8)
            for row in self.trace.lower[-ticks:]:
                np.testing.assert_allclose(row['executed_rpm'], row['commanded_rpm_clipped'], rtol=1e-6)
            self.rows.append(dict(next_state=next_state.copy(), replay_action=trainer.buffer.action[index].copy(),
                                  replay_reward=trainer.buffer.reward[index, 0],
                                  terminated=terminated, truncated=truncated, lower_ticks=ticks,
                                  packet_positions=env.last_reference_packet.positions.copy(),
                                  packet_velocities=env.last_reference_packet.velocities.copy()))
        if self.pool_mode:
            record = {**env._pool_info(), 'step': trainer.total_steps, 'start_level': env._start_level,
                      'start_source': env._start_source, 'reward': reward, 'replay_reward': reward * trainer.args.reward_scale,
                      'terminated': bool(terminated), 'truncated': bool(truncated), 'done_reason': info['done_reason'],
                      'goal_distance': float(info['goal_distance']), 'lower_ticks': ticks,
                      'minimum_tick_clearance': min(r['clearance'] for r in self.trace.lower[-ticks:]),
                      'command_velocity': env.command_velocity.tolist(), 'actual_velocity': env.vel[0].tolist(),
                      'state_abs_max': float(np.max(np.abs(next_state)))}
            with (self.root / 'transitions.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            if terminated or truncated:
                episode = {**record, 'length': len(self.trace.upper),
                           'return': sum(r['reward'] for r in self.trace.upper),
                           'initial_goal_distance': float(np.linalg.norm(self.trace.initial_position-self.trace.target)),
                           'minimum_goal_distance': min([float(np.linalg.norm(self.trace.initial_position-self.trace.target))] +
                                                        [r['goal_distance'] for r in self.trace.lower]),
                           'episode_minimum_tree_clearance': min(r['clearance'] for r in self.trace.lower),
                           'reward_terms': {key: float(trainer._episode_reward_term_sums.get(key, 0.) + info.get('reward/' + key, 0.))
                                            for key in trainer._episode_reward_term_sums}}
                with (self.root / 'episodes.jsonl').open('a') as f:
                    f.write(json.dumps(episode) + '\n')
        if terminated or truncated:
            self.finish(info)

    def finish(self, info=None):
        if self.trace is None:
            return
        trace = self.trace
        trace.uninstall()
        if trace.upper:
            info = info or {'done_reason': 'p0_budget_boundary', 'success': False}
            if self.pool_mode and self.trainer.total_steps > self.trainer.args.scratch_audit_trace_steps and info['done_reason'] in self.saved_reasons:
                self.trace = None
                return
            self.saved_reasons.add(info['done_reason'])
            trace.save(self.root / 'traces', done_reason=info['done_reason'],
                       success=info.get('success', False), final_goal_distance=trace.upper[-1]['goal_distance'])
            np.savez_compressed(self.root / 'traces' / f'{trace.trace_id}_replay.npz',
                                **{k: np.asarray([r[k] for r in self.rows]) for k in self.rows[0]})
        self.trace = None

    def on_step(self):
        t = self.trainer
        if t.last_train_info is not None:
            assert all(v is None or np.isfinite(v) for v in t.last_train_info.values())
            with (self.root / 'updates.jsonl').open('a') as f:
                f.write(json.dumps({'step': t.total_steps, **t.last_train_info}) + '\n')
        if t.total_steps % t.args.ckpt_interval == 0:
            self.monitor_policy()
            save_training(t)

    def monitor_policy(self):
        t = self.trainer
        if not t.buffer.size:
            return
        previous = None
        if self.probe_path.exists():
            with np.load(self.probe_path) as data:
                states, actions, previous = data['states'], data['actions'], data['policy']
        else:
            states = t.buffer.state[:min(256, t.buffer.size)].copy()
            actions = t.buffer.action[:len(states)].copy()
        metrics, policy = policy_diagnostics(t.agent, states, actions, t.env.TARGET_POS)
        if self.pool_mode:
            indices = (t.buffer.ptr - 1 - np.arange(min(256, t.buffer.size))) % t.buffer.max_size
            current_states = t.buffer.state[indices]
            current_metrics, _ = policy_diagnostics(t.agent, current_states, t.buffer.action[indices], t.env.TARGET_POS)
            metrics.update({f'visited/{k}': v for k, v in current_metrics.items()})
            distances = np.linalg.norm(world_goal_delta(current_states, t.env.TARGET_POS), axis=1)
            for name, mask in (('near', distances < 1.2), ('far', distances >= 1.2)):
                metrics[f'{name}/probe_count'] = int(mask.sum())
                if mask.any():
                    group, _ = policy_diagnostics(t.agent, current_states[mask], t.buffer.action[indices][mask], t.env.TARGET_POS)
                    metrics.update({f'{name}/{k}': v for k, v in group.items()})
        metrics['adjacent_policy_rms_change'] = (
            float(np.sqrt(np.mean((policy - previous) ** 2))) if previous is not None else None)
        np.savez_compressed(self.probe_path, states=states, actions=actions, policy=policy)
        with (self.root / 'policy_diagnostics.jsonl').open('a') as handle:
            handle.write(json.dumps({'step': t.total_steps, **metrics}) + '\n')

    def on_end(self):
        self.finish()
        self.monitor_policy()
        self.trainer.agent.save(str(Path(self.trainer.args.ckpt_dir) / 'model_raw_final'))
        save_training(self.trainer)
        if self.pool_mode:
            self.trainer.agent.save(str(Path(self.trainer.args.ckpt_dir) / f'model_raw_step{self.trainer.total_steps}'))
            for callback in self.trainer.callbacks:
                if type(callback).__name__ == 'EgoPoolEvalCallback':
                    callback.on_step(self.trainer, force=True, context='final_raw')
