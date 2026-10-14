"""Extend the exact Full seed43 4k critic state to cumulative 20k transitions.

The source actor (MLP/GRU) and its target remain immutable for the whole run.
Learning uses the original TD3 implementation, LR=3e-4 and original replay,
smoothing, exploration, reward, stage mixer and control chain.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

from scripts import change_td3
from scripts.run_change_td3 import build_mode_cmd
from scripts.run_full_ft_v2 import CHECK_FIELDS, differences, dump, sha, protected_hashes
from scripts.reproduce_full_ft_v1_unfreeze import SnapshotObserver, buffer_payload, original_tape, preserve_rng
from trainers.td3_trainer import TD3Trainer
from trainers.callbacks.checkpoint import BestCheckpointCallback, ConvergenceGuardCallback
from scripts.audit_full_ft_v1_unfreeze import td_statistics

DEFAULT_ROOT = Path('analysis_outputs/full_critic_frozen_v1_seed43')
DEFAULT_CKPTS = Path('checkpoints/full_critic_frozen_v1_seed43')
DEFAULT_RUNS = Path('runs/full_critic_frozen_v1_seed43')
REFERENCE = Path('checkpoints/full_ft_v1_unfreeze_audit_20261012/seed43/model_004000_full_snapshot.pt')
SAVE_STEPS = (4000, 5000, 10000, 15000, 20000)


def production_hashes():
    files = {p for folder in ('envs', 'async_control', 'algos', 'trainers', 'high_level_planner', 'models', 'data', 'utils')
             for p in Path(folder).rglob('*.py')}
    files.update(Path(p) for p in ('data/map_atlas/catalog_v1.json', 'scripts/change_td3.py',
                                  'scripts/run_change_td3.py', 'scripts/audit_stage0_feasibility.py',
                                  'scripts/audit_full_ft_v1_critic_actor.py'))
    return {str(p): sha(p) for p in sorted(files)}


def freeze_actor(agent):
    for module in (agent.actor, agent.actor_target):
        module.requires_grad_(False)
    agent.actor_frozen = True
    if not all(torch.equal(v, agent.actor_target.state_dict()[k]) for k, v in agent.actor.state_dict().items()):
        raise RuntimeError('Actor target must initially equal the source actor')
    def forbidden_step(*args, **kwargs):
        raise RuntimeError('Frozen-actor experiment forbids actor optimizer.step()')
    agent.actor_optimizer.step = forbidden_step


def check_learner(agent, source, expected):
    for name in ('actor', 'actor_target'):
        if differences(getattr(agent, name).state_dict(), source, name):
            raise RuntimeError(f'{name} differs from the fixed source')
        if any(p.requires_grad or p.grad is not None for p in getattr(agent, name).parameters()):
            raise RuntimeError(f'{name} has trainable parameters or gradients')
    if agent.actor_optimizer.state:
        raise RuntimeError('Actor Adam must remain empty')
    if not agent.actor_frozen:
        raise RuntimeError('Actor update flag was enabled')
    if any(g['lr'] != 3e-4 for g in agent.critic_optimizer.param_groups):
        raise RuntimeError('Critic LR must stay exactly 3e-4')
    for name, key in [('discount', 'gamma'), ('tau', 'tau'), ('policy_freq', 'policy_freq'),
                      ('policy_noise', 'policy_noise'), ('noise_clip', 'noise_clip')]:
        if getattr(agent, name) != expected[key]:
            raise RuntimeError(f'Original TD3 {name} changed')
    if not all(p.requires_grad for p in agent.critic.parameters()):
        raise RuntimeError('Both critic GRUs and MLPs must remain trainable')


class FrozenCriticTrainer(TD3Trainer):
    options = None

    def add_callback(self, callback):
        if isinstance(callback, BestCheckpointCallback):
            # Disable it before change_td3 invokes the initial baseline selector.
            callback.on_step = lambda trainer: None
            callback.on_train_end = lambda trainer: None
        super().add_callback(callback)

    def __init__(self, env, agent, args):
        super().__init__(env, agent, args)
        self.root = self.options.output_root
        self.ckpts = self.options.checkpoint_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.observer = SnapshotObserver(self, self.root, self.ckpts)
        self.source = copy.deepcopy(agent.actor.state_dict())
        self.expected = torch.load(REFERENCE, map_location='cpu', weights_only=False)
        freeze_actor(agent)
        self.add_callback(self)
        self.prefix_verified = False
        self.frozen_checks = []
        self.original = original_tape(43)
        self.prefix_errors = {'steps_checked': 0, 'state': 0., 'action': 0., 'reward': 0., 'position': 0.}
        self.initial_mismatches = []
        for name in ('actor', 'critic'):
            initial = torch.load(f'checkpoints/full_ft_v1_seed43/model_initial_{name}', map_location='cpu', weights_only=True)
            self.initial_mismatches += differences(getattr(agent, name).state_dict(), initial, name)
        if self.initial_mismatches:
            raise RuntimeError(f'Initial networks differ: {self.initial_mismatches}')
        self.probe_indices = np.sort(np.random.default_rng(43 + 20261012).choice(4000, 512, replace=False))
        self.fixed_sample = {k: self.expected['replay'][k][self.probe_indices]
                             for k in ('state', 'action', 'next_state', 'reward', 'not_done')}
        self.probes = []
        original_step = env.step
        def observed_step(action):
            state = self.state.copy()
            result = original_step(action)
            if self.total_steps <= 4000:
                i = self.total_steps - 1
                errors = {'state': float(np.abs(state - self.original['state'][i]).max()),
                          'action': float(np.abs(np.asarray(action).reshape(3) - self.original['action'][i]).max()),
                          'reward': float(abs(result[1] - self.original['reward'][i])),
                          'position': float(np.abs(env.pos[0] - self.original['position'][i]).max())}
                self.prefix_errors['steps_checked'] += 1
                for k, v in errors.items():
                    self.prefix_errors[k] = max(self.prefix_errors[k], v)
                if any(v != 0 for v in errors.values()):
                    dump(self.root / 'prefix_failure.json', {'step': self.total_steps, 'differences': errors})
                    raise RuntimeError(f'Original training prefix differs at step {self.total_steps}')
            return result
        env.step = observed_step

    def save_snapshot(self):
        with preserve_rng():
            self.observer.save(self.total_steps)
            path = self.ckpts / f'model_{self.total_steps:06d}_full_snapshot.pt'
            p = torch.load(path, map_location='cpu', weights_only=False)
            if not p['replay_included']:
                p['replay'] = buffer_payload(self.buffer)
                p['success_replay'] = {k: buffer_payload(b) for k, b in self.success_buffer.buffers.items()}
            p['replay_included'] = True
            p['experiment'] = 'full_critic_frozen_v1'
            p['new_transitions_after_4k'] = max(0, self.total_steps - 4000)
            p['actor_parameters_require_grad'] = False
            p['actor_target_parameters_require_grad'] = False
            p['critic_lr_fixed'] = 3e-4
            p['last_train_info'] = self.last_train_info
            torch.save(p, path)
        return p

    def verify_prefix(self, p):
        mismatches = [d for k in CHECK_FIELDS for d in differences(p[k], self.expected[k], k)]
        dump(self.root / 'prefix_verification.json', {
            'reference': str(REFERENCE), 'reference_sha256': sha(REFERENCE),
            'all_fields_exactly_equal': not mismatches, 'mismatches': mismatches,
            'fields_checked': CHECK_FIELDS, 'trajectory': self.prefix_errors,
            'torch_threads': torch.get_num_threads(),
            'seed43_prefix_replay_exception_authorized_in_previous_turn': True})
        if mismatches or torch.get_num_threads() != 6:
            raise RuntimeError(f'Cannot continue incomparable 4k state: {mismatches[:20]}')
        self.prefix_verified = True
        print('[Frozen critic] exact complete 4k state verified; collecting 16000 new transitions for 20k main budget', flush=True)

    def offline_probe(self, payload):
        with preserve_rng():
            fixed = td_statistics(self.agent, self.fixed_sample, torch.Generator().manual_seed(43 + 726))
            replay = payload['replay']
            terminal = replay['not_done'][:, 0] == 0
            records = {}
            for name, mask in [('success', terminal & (replay['reward'][:, 0] >= 20)),
                               ('failure', terminal & (replay['reward'][:, 0] < 0))]:
                n = int(mask.sum())
                if n:
                    with torch.no_grad():
                        q1, q2 = self.agent.critic(replay['state'][mask], replay['action'][mask])
                    records[name] = {'transitions': n, 'mean_scaled_reward': float(replay['reward'][mask].mean()),
                                     'q1_rmse': float((q1 - replay['reward'][mask]).square().mean().sqrt()),
                                     'q2_rmse': float((q2 - replay['reward'][mask]).square().mean().sqrt())}
                else:
                    records[name] = {'transitions': 0}
            row = {'step': self.total_steps, 'critic_updates': self.agent.total_it,
                   'actor_updates': 0, 'main_replay_size': self.buffer.size,
                   'success_replay_size': self.success_buffer.size,
                   'success_replay_strata': self.success_buffer.strata_sizes,
                   'fixed_4k_td_probe': fixed, 'terminal_fit': records,
                   'training_episode_index': self._full_ft_episode_index,
                   'stage_counts': self.full_ft_stage_mixer.stage_counts.copy(),
                   'diagnostic_role': 'training fit only; not MC value or ranking acceptance'}
            self.probes.append(row)
            dump(self.root / 'critic_probes.json', self.probes)

    def on_step(self, trainer):
        info = self.last_train_info or {}
        if info.get('actor_updated') or info.get('actor_frozen', 1.) != 1.:
            raise RuntimeError('Actor update attempted during frozen experiment')
        if any(g['lr'] != 3e-4 for g in self.agent.critic_optimizer.param_groups):
            raise RuntimeError('Critic LR changed')
        if self.total_steps % 1000 == 0 or self.total_steps == self.args.total_steps:
            check_learner(self.agent, self.source, self.expected['args'])
            self.frozen_checks.append({'step': self.total_steps, 'actor_equal_source': True,
                                       'actor_target_equal_source': True, 'actor_updates': 0,
                                       'actor_adam_empty': True, 'critic_lr': 3e-4})
            dump(self.root / 'frozen_checks.json', self.frozen_checks)
        if self.total_steps in SAVE_STEPS or self.total_steps == self.args.total_steps:
            payload = self.save_snapshot()
            if self.total_steps == 4000:
                self.verify_prefix(payload)
            if self.total_steps >= 4000:
                self.offline_probe(payload)
        if info and not np.isfinite(info['critic_loss']):
            raise RuntimeError('Nonfinite critic loss: experiment stopped, no replacement run')

    def on_episode_end(self, trainer):
        pass

    def on_train_end(self, trainer):
        check_learner(self.agent, self.source, self.expected['args'])
        dump(self.root / 'result.json', {
            'requested_budget_completed': self.total_steps == self.args.total_steps,
            'main_20k_completed': self.total_steps == 20000,
            'cumulative_upper_transitions': self.total_steps,
            'new_transitions_after_4k': self.total_steps - 4000,
            'critic_updates': self.agent.total_it, 'actor_updates': 0,
            'prefix_verified': self.prefix_verified, 'critic_lr': 3e-4,
            'checkpoint_directory': str(self.ckpts), 'no_best_critic_restore': True})

    def run(self):
        # Navigation success evaluates the fixed policy, not critic accuracy.
        # Therefore it must never select/restore an earlier critic checkpoint.
        self.callbacks = [cb for cb in self.callbacks
                          if not isinstance(cb, (BestCheckpointCallback, ConvergenceGuardCallback)) and cb is not self]
        self.callbacks.append(self)
        check_learner(self.agent, self.source, self.expected['args'])
        dump(self.root / 'configuration.json', {
            'experiment': 'full_critic_frozen_v1', 'seed': 43,
            'cumulative_target': self.args.total_steps, 'starting_snapshot_step': 4000,
            'expected_new_transitions': self.args.total_steps - 4000,
            'actor_mlp_gru_frozen': True, 'actor_target_equal_source': True,
            'critic_lr': 3e-4, 'checkpoint_directory': str(self.ckpts),
            'log_directory': self.args.log_dir, 'args': vars(self.args),
            'audit_steps': [4000, 10000, 20000],
            'best_critic_selection_and_restore': False,
            'production_inputs': production_hashes(),
            'source_sha256': sha(Path(self.args.full_ft_actor_source + '_actor'))})
        try:
            super().run()
        except BaseException as exc:
            dump(self.root / 'failure.json', {'step': self.total_steps, 'error': str(exc),
                                             'prefix_verified': self.prefix_verified})
            raise


def build_command(cli):
    command = build_mode_cmd('full_ft_v1')[3:]
    options = {'--seed': 43, '--total_steps': cli.total_steps,
               '--full_ft_actor_freeze_steps': 20000,
               '--upper_critic_lr': '3e-4',
               '--log_dir': cli.log_dir, '--ckpt_dir': cli.checkpoint_dir,
               '--full_ft_trace_dir': cli.output_root / 'training_traces',
               '--trajectory_output_dir': cli.output_root / 'trajectory',
               '--collision_diagnostic_output_dir': cli.output_root / 'collisions',
               '--timeout_diagnostic_output_dir': cli.output_root / 'timeouts',
               '--eval_terminal_coverage_output_dir': cli.output_root / 'coverage'}
    for option, value in options.items():
        if option in command:
            command[command.index(option) + 1] = str(value)
        else:
            command += [option, str(value)]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--total_steps', type=int, default=20000, help='CUMULATIVE upper transitions, including the exact 4k prefix')
    parser.add_argument('--output_root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--checkpoint_dir', type=Path, default=DEFAULT_CKPTS)
    parser.add_argument('--log_dir', type=Path, default=DEFAULT_RUNS)
    parser.add_argument('--audit', action='store_true', help='After 20k, run both acceptance levels and target-noise control')
    cli = parser.parse_args()
    if not 4000 <= cli.total_steps <= 20000:
        parser.error('Budget must be cumulative 4000..20000; use 20000 for the main experiment')
    if cli.audit and cli.total_steps != 20000:
        parser.error('--audit requires the complete cumulative 20k main experiment')
    for folder in (cli.output_root, cli.checkpoint_dir, cli.log_dir):
        if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
            parser.error(f'Nonempty output path {folder}; refusing to overwrite or pretend to live-resume it')
    source = Path('checkpoints/startobscurr_v1_selected_20261010/seed43/model_final_actor')
    manifest = json.loads((source.parent.parent / 'manifest.json').read_text())
    if sha(source) != manifest['seeds']['43']['files']['model_final_actor']['sha256']:
        raise RuntimeError('Frozen source actor hash differs from manifest')
    torch.set_num_threads(6)
    FrozenCriticTrainer.options = cli
    before = protected_hashes()
    dump(cli.output_root / 'original_artifact_hashes_before.json', before)
    with patch.object(change_td3, 'TD3Trainer', FrozenCriticTrainer), patch.object(sys, 'argv', [__file__] + build_command(cli)):
        change_td3.main()
    after = protected_hashes()
    changed = [p for p in before if after.get(p) != before[p]]
    dump(cli.output_root / 'original_artifact_integrity.json', {'protected_files': len(before), 'changed_files': changed,
                                                             'all_original_files_unchanged': not changed})
    if changed:
        raise RuntimeError(f'Original artifacts changed: {changed[:10]}')
    if cli.audit:
        from scripts.audit_full_critic_frozen import audit_all
        audit_all(cli.output_root)


if __name__ == '__main__':
    main()
