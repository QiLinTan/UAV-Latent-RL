"""Instrument the existing Full FT v1 trainer for a 5000-step diagnostic rerun.

All outputs are isolated. Learning, sampling, warmup, noise and the original
validation callbacks are unchanged. Observer work preserves global RNG state.
Original 5000-step weights and logged trajectories decide reproducibility.
"""

import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import random
import sys
from unittest.mock import patch

import numpy as np
import torch

from scripts import change_td3
from scripts.run_change_td3 import build_mode_cmd
from trainers.td3_trainer import TD3Trainer

ROOT = Path('analysis_outputs/full_ft_v1_unfreeze_audit_20261012')
CKPTS = Path('checkpoints/full_ft_v1_unfreeze_audit_20261012')
STEPS = (0, 3999, 4000, 4001, 4019, 4500, 5000)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def rng_state():
    return {'numpy': np.random.get_state(), 'python': random.getstate(),
            'torch_cpu': torch.get_rng_state().clone()}


@contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        np.random.set_state(state['numpy']); random.setstate(state['python'])
        torch.set_rng_state(state['torch_cpu'])


def buffer_payload(buffer):
    return {'size': int(buffer.size), 'ptr': int(buffer.ptr), 'capacity': int(buffer.max_size),
            **{key: torch.from_numpy(getattr(buffer, key)[:buffer.size].copy())
               for key in ('state', 'action', 'next_state', 'reward', 'not_done')}}


def parameter_comparison(actual, expected):
    differences = [(actual[key].detach().cpu() - expected[key]).abs().max().item() for key in expected]
    return {'all_tensors_equal': all(torch.equal(actual[key].detach().cpu(), expected[key]) for key in expected),
            'maximum_absolute_difference': max(differences)}


def original_tape(seed):
    states, actions, rewards, positions = [], [], [], []
    for path in sorted(Path(f'analysis_outputs/full_ft_v1_seed{seed}/training_traces').glob('episode_*/*.json')):
        meta = json.loads(path.read_text()); first = meta['global_upper_step'] - meta['upper_steps'] + 1
        if first > 5000:
            break
        n = min(meta['upper_steps'], 5001 - first)
        with np.load(path.with_suffix('.npz')) as data:
            states.append(data['upper_pre_observation'][:n]); actions.append(data['upper_action'][:n])
            rewards.append(data['upper_reward'][:n]); positions.append(data['upper_position'][:n])
    return {key: np.concatenate(values) for key, values in
            (('state', states), ('action', actions), ('reward', rewards), ('position', positions))}


class SnapshotObserver:
    def __init__(self, trainer, output, checkpoint_dir):
        self.trainer, self.output, self.checkpoint_dir = trainer, output, checkpoint_dir
        self.actor_updates = 0

    def save(self, step):
        trainer, agent = self.trainer, self.trainer.agent
        with preserve_rng():
            prefix = self.checkpoint_dir / f'model_{step:06d}'
            prefix.parent.mkdir(parents=True, exist_ok=True)
            agent.save(str(prefix))
            include_replay = step in (0, 4000, 5000)
            env = trainer.env; base = env.base_env
            fields = ('runtime_lower_step', 'reference_position', 'reference_velocity',
                      'reference_acceleration', 'command_velocity', 'last_reference_packet',
                      'last_reference_sample', 'last_motor_action', 'packet_generation_count', '_semantic_history')
            base_fields = ('step_counter', '_prev_pos', '_prev_goal_dist', '_reset_count',
                           'pos', 'vel', 'rpy', 'ang_v', '_last_motor_rpm')
            payload = {
                'schema_version': 1, 'upper_step': step, 'actor_updates': self.actor_updates,
                'critic_updates': agent.total_it, 'actor_frozen': agent.actor_frozen,
                'actor': agent.actor.state_dict(), 'actor_target': agent.actor_target.state_dict(),
                'critic': agent.critic.state_dict(), 'critic_target': agent.critic_target.state_dict(),
                'actor_optimizer': agent.actor_optimizer.state_dict(), 'critic_optimizer': agent.critic_optimizer.state_dict(),
                'rng': rng_state(), 'args': vars(trainer.args),
                'trainer': {'state': trainer.state.copy(), 'episode_step': trainer.episode_step,
                            'episode_return': float(trainer.episode_return), 'episode_index': trainer._full_ft_episode_index,
                            'episode_pending_success_transitions': trainer._episode_success_transitions},
                'stage_mixer': copy.deepcopy(trainer.full_ft_stage_mixer.__dict__),
                'environment': {key: copy.deepcopy(getattr(env, key)) for key in fields},
                'base_environment': {key: copy.deepcopy(getattr(base, key)) for key in base_fields},
                'environment_rng': copy.deepcopy(base._rng.bit_generator.state),
                'controller': copy.deepcopy(env.teacher.__dict__),
                'tree_specs': copy.deepcopy(base._tree_specs), 'target': np.asarray(env.TARGET_POS).copy(),
                'replay_included': include_replay,
                'replay_sizes': {'main': trainer.buffer.size, 'success': trainer.success_buffer.size},
                'simulation_live_resume_supported': False,
            }
            if include_replay:
                payload['replay'] = buffer_payload(trainer.buffer)
                payload['success_replay'] = {level: buffer_payload(buffer) for level, buffer in trainer.success_buffer.buffers.items()}
            torch.save(payload, str(prefix) + '_full_snapshot.pt')
            dump(Path(str(prefix) + '_metadata.json'), {
                'upper_step': step, 'actor_updates': self.actor_updates, 'critic_updates': agent.total_it,
                'actor_frozen': bool(agent.actor_frozen), 'replay_included': include_replay,
                'replay_main_size': trainer.buffer.size, 'success_replay_size': trainer.success_buffer.size,
                'torch_threads': torch.get_num_threads(), 'reward_integration_dt': base._reward_model.integration_dt,
                'actor_lr': agent.actor_optimizer.param_groups[0]['lr'],
                'critic_lr': agent.critic_optimizer.param_groups[0]['lr'],
            })
            print(f'[Audit snapshot] seed={trainer.args.seed} step={step} actor_updates={self.actor_updates}', flush=True)

    def on_step(self, trainer):
        if trainer.last_train_info and trainer.last_train_info['actor_updated']:
            self.actor_updates += 1
        if trainer.total_steps in STEPS:
            self.save(trainer.total_steps)

    def on_episode_end(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass


class DiagnosticTrainer(TD3Trainer):
    def __init__(self, env, agent, args):
        super().__init__(env, agent, args)
        self.audit_root = ROOT / f'seed{args.seed}'
        self.audit_root.mkdir(parents=True, exist_ok=True)
        self.observer = SnapshotObserver(self, self.audit_root, CKPTS / f'seed{args.seed}')
        self.add_callback(self.observer)
        self.original = original_tape(args.seed)
        self.trajectory_comparison = {'steps_checked': 0, 'max_state_abs_diff': 0., 'max_action_abs_diff': 0.,
                                     'max_reward_abs_diff': 0., 'max_position_abs_diff': 0., 'first_difference_step': None}
        self.reward_check = {'upper_transitions': 0, 'lower_ticks': 0, 'max_lower_sum_error': 0.,
                             'max_time_cost_error': 0., 'max_reward_identity_error': 0.}
        self.initial_comparison = {}
        for name, module in (('actor', agent.actor), ('critic', agent.critic)):
            expected = torch.load(f'checkpoints/full_ft_v1_seed{args.seed}/model_initial_{name}', map_location='cpu', weights_only=True)
            self.initial_comparison[name] = parameter_comparison(module.state_dict(), expected)
        self.update_rows = []
        fixed = np.load(f'analysis_outputs/full_ft_v1_critic_actor_audit_20261012/seed{args.seed}_warmup_bank.npz')
        self.fixed_states = torch.as_tensor(fixed['observations'].copy(), dtype=torch.float32, device=agent.device)
        original_train = agent.train
        def observed_train(*positional, **keyword):
            scheduled = (agent.total_it + 1) % agent.policy_freq == 0 and not agent.actor_frozen
            capture = scheduled and self.observer.actor_updates < 10
            before = {key: value.detach().clone() for key, value in agent.actor.state_dict().items()} if capture else None
            if capture:
                with preserve_rng(), torch.no_grad():
                    before_actions = agent.actor(self.fixed_states).cpu().numpy()
            info = original_train(*positional, **keyword)
            if capture:
                with preserve_rng(), torch.no_grad():
                    after_actions = agent.actor(self.fixed_states).cpu().numpy()
                old = torch.cat([value.flatten() for value in before.values()])
                new = torch.cat([value.detach().flatten() for value in agent.actor.state_dict().values()])
                row = {'upper_step': self.total_steps, 'actor_update_index': self.observer.actor_updates + 1,
                       'actor_grad_norm_before_clip': info['actor_grad_norm'],
                       'actor_grad_norm_after_clip': float(torch.sqrt(sum(p.grad.detach().square().sum() for p in agent.actor.parameters()))),
                       'parameter_delta_l2': float(torch.linalg.vector_norm(new - old)),
                       'relative_parameter_delta_l2': float(torch.linalg.vector_norm(new - old) / torch.linalg.vector_norm(old)),
                       'maximum_parameter_change': float((new - old).abs().max()),
                       'fixed_bank_mean_velocity_change_mps': float(np.mean(np.linalg.norm((after_actions - before_actions) * [.8, .8, .3], axis=1))),
                       'actor_loss': info['actor_loss'], 'critic_loss': info['critic_loss']}
                self.update_rows.append(row)
                dump(self.audit_root / 'first_actor_updates.json', self.update_rows)
            return info
        agent.train = observed_train
        upper_step = env.step
        def observed_upper(action):
            trace = self._full_ft_trace
            start = len(trace.lower)
            before_state = self.state.copy()
            result = upper_step(action)
            lower = trace.lower[start:]
            reward, terminated, truncated, info = result[1:]
            lower_sum = sum(row['reward'] for row in lower)
            running_ticks = len(lower) - int(terminated or truncated)
            expected_time = args.time_penalty_weight * running_ticks / args.ctrl_freq
            identity = info['reward/progress_reward'] - info['reward/continuous_penalty_total'] + info['reward/terminal_outcome_reward']
            self.reward_check['upper_transitions'] += 1; self.reward_check['lower_ticks'] += len(lower)
            for name, error in (('max_lower_sum_error', abs(reward - lower_sum)),
                                ('max_time_cost_error', abs(info['reward/time_penalty'] - expected_time)),
                                ('max_reward_identity_error', abs(reward - identity))):
                self.reward_check[name] = max(self.reward_check[name], float(error))
            index = self.total_steps - 1
            errors = {'max_state_abs_diff': float(np.max(np.abs(before_state - self.original['state'][index]))),
                      'max_action_abs_diff': float(np.max(np.abs(np.asarray(action).reshape(3) - self.original['action'][index]))),
                      'max_reward_abs_diff': float(abs(reward - self.original['reward'][index])),
                      'max_position_abs_diff': float(np.max(np.abs(env.pos[0] - self.original['position'][index])))}
            self.trajectory_comparison['steps_checked'] += 1
            for key, value in errors.items():
                self.trajectory_comparison[key] = max(self.trajectory_comparison[key], value)
            if self.trajectory_comparison['first_difference_step'] is None and max(errors.values()) > 1e-5:
                self.trajectory_comparison['first_difference_step'] = self.total_steps
            return result
        env.step = observed_upper

    def run(self):
        self.observer.save(0)
        super().run()
        comparisons = {}
        for name in ('actor', 'critic'):
            actual = torch.load(CKPTS / f'seed{self.args.seed}' / f'model_005000_{name}', map_location='cpu', weights_only=True)
            expected = torch.load(f'checkpoints/full_ft_v1_seed{self.args.seed}/model_5000_{name}', map_location='cpu', weights_only=True)
            comparisons[name] = parameter_comparison(actual, expected)
        assert self.reward_check['max_lower_sum_error'] < 1e-8
        assert self.reward_check['max_time_cost_error'] < 1e-8
        assert self.reward_check['max_reward_identity_error'] < 1e-5
        summary = {'seed': self.args.seed, 'upper_steps': self.total_steps,
                   'initial_comparison': self.initial_comparison, 'final_5000_comparison': comparisons,
                   'exact_original_weight_reproduction': all(row['all_tensors_equal'] for row in comparisons.values()),
                   'trajectory_comparison': self.trajectory_comparison, 'formal_training_reward_aggregation': self.reward_check,
                   'snapshot_steps': list(STEPS), 'first_actor_updates': self.update_rows,
                   'torch_threads': torch.get_num_threads()}
        dump(self.audit_root / 'reproduction_summary.json', summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(43, 45), required=True)
    cli = parser.parse_args()
    output = ROOT / f'seed{cli.seed}'
    if (output / 'reproduction_summary.json').exists():
        print((output / 'reproduction_summary.json').read_text()); return
    command = build_mode_cmd('full_ft_v1')[3:]
    options = {
        '--seed': cli.seed, '--total_steps': 5000,
        '--full_ft_actor_source': f'checkpoints/startobscurr_v1_selected_20261010/seed{cli.seed}/model_final',
        '--log_dir': output / 'runs', '--ckpt_dir': output / 'ordinary_checkpoints',
        '--full_ft_trace_dir': output / 'training_traces',
        '--trajectory_output_dir': output / 'trajectory', '--collision_diagnostic_output_dir': output / 'collisions',
        '--timeout_diagnostic_output_dir': output / 'timeouts', '--eval_terminal_coverage_output_dir': output / 'coverage',
    }
    for name, value in options.items():
        command[command.index(name) + 1] = str(value)
    with patch.object(change_td3, 'TD3Trainer', DiagnosticTrainer), patch.object(sys, 'argv', [__file__] + command):
        change_td3.main()


if __name__ == '__main__':
    main()
