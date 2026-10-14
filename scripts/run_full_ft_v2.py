"""Isolated conservative Full TD3 ablation with fail-closed prefix verification.

The original production trainer/train method execute every learning step. Each
branch replays the authorized seed43 prefix and must match the existing complete
4k snapshot before its two optimizer LRs and fixed source anchor are configured.
"""
import argparse
from collections import deque
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch

from scripts import change_td3
from scripts.run_change_td3 import build_mode_cmd
from scripts.reproduce_full_ft_v1_unfreeze import (
    SnapshotObserver, dump as strict_dump, original_tape, preserve_rng,
)
from scripts.audit_full_ft_v1_critic_actor import actor_batch, command_metrics
from trainers.callbacks.checkpoint import BestCheckpointCallback
from trainers.callbacks.eval_callback import FullFTTripleEvalCallback
from trainers.td3_trainer import TD3Trainer

ROOT = Path('analysis_outputs/full_ft_v2_seed43')
CKPTS = Path('checkpoints/full_ft_v2_seed43')
REFERENCE = Path('checkpoints/full_ft_v1_unfreeze_audit_20261012/seed43/model_004000_full_snapshot.pt')
BRANCHES = {'A': (1e-4, 0.), 'B': (3e-4, .1), 'C': (1e-4, .1)}
UPDATE_PROBES = (1, 10, 50, 100, 500)
CHECK_FIELDS = ('upper_step', 'actor_updates', 'critic_updates', 'actor_frozen',
                'actor', 'actor_target', 'critic', 'critic_target',
                'actor_optimizer', 'critic_optimizer', 'rng', 'trainer',
                'stage_mixer', 'environment', 'base_environment',
                'environment_rng', 'controller', 'tree_specs', 'target',
                'replay', 'success_replay')


def json_values(value):
    """Keep undefined K1 tracking diagnostics explicit in standard JSON."""
    if isinstance(value, dict):
        return {k: json_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_values(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def dump(path, value):
    strict_dump(path, json_values(value))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_hashes():
    paths = set()
    for parent in (Path('checkpoints'), Path('analysis_outputs'), Path('runs')):
        for folder in parent.glob('full_ft_v1*'):
            paths.update(p for p in folder.rglob('*') if p.is_file())
    frozen = Path('checkpoints/startobscurr_v1_selected_20261010')
    paths.update(p for p in frozen.rglob('*') if p.is_file())
    return {str(p): sha(p) for p in sorted(paths)}


def differences(a, b, path='root'):
    """Exact recursive comparison, including arrays in packets and RNGs."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return [] if torch.equal(a.cpu(), b.cpu()) else [path]
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return [] if np.array_equal(a, b, equal_nan=True) else [path]
    if isinstance(a, np.random.Generator) and isinstance(b, np.random.Generator):
        return differences(a.bit_generator.state, b.bit_generator.state, path)
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [path + '.keys']
        return [d for k in a for d in differences(a[k], b[k], f'{path}.{k}')]
    if isinstance(a, (list, tuple, deque)) and isinstance(b, (list, tuple, deque)):
        if len(a) != len(b):
            return [path + '.length']
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f'{path}[{i}]')]
    if type(a) is type(b) and hasattr(a, '__dict__') and not isinstance(a, type):
        # Enum members have internal class references; equality is their contract.
        from enum import Enum
        if not isinstance(a, Enum):
            return differences(vars(a), vars(b), path)
    return [] if type(a) is type(b) and a == b else [path]


def fully_collapsed(row):
    return all(row[name]['summary']['eval/success_rate'] == 0.
               for name in ('full_stage0', 'full_stage3', 'k1_stage3'))


def stable_for_extension(row):
    thresholds = {'full_stage0': .8, 'full_stage3': .7, 'k1_stage3': .8}
    return all(row[name]['summary']['eval/success_rate'] >= threshold
               for name, threshold in thresholds.items())


class AblationTrainer(TD3Trainer):
    branch = None

    def __init__(self, env, agent, args):
        super().__init__(env, agent, args)
        self.root = ROOT / self.branch
        self.ckpts = CKPTS / self.branch
        self.root.mkdir(parents=True, exist_ok=True)
        self.observer = SnapshotObserver(self, self.root, self.ckpts)
        self.add_callback(self)
        self.prefix_verified = False
        self.original = original_tape(43)
        self.prefix_errors = dict(steps_checked=0, maximum_state_error=0., maximum_action_error=0.,
                                  maximum_reward_error=0., maximum_position_error=0.)
        self.initial_differences = []
        for name in ('actor', 'critic'):
            reference = torch.load(f'checkpoints/full_ft_v1_seed43/model_initial_{name}',
                                   map_location='cpu', weights_only=True)
            self.initial_differences += differences(getattr(agent, name).state_dict(), reference, name)
        if self.initial_differences:
            raise RuntimeError(f'Initial model mismatch: {self.initial_differences}')
        with np.load('analysis_outputs/full_ft_v1_critic_actor_audit_20261012/seed43_warmup_bank.npz') as bank:
            self.bank = {k: bank[k].copy() for k in bank.files}
        self.source_weights = copy.deepcopy(agent.actor.state_dict())
        with preserve_rng():
            self.source_actions = actor_batch(agent, self.bank['observations'])
        self.diagnostics = []
        self.evaluations = []
        self.training_episodes = []
        self.anchor_loss_updates = []
        self.extension_decision = None
        self.stop_decision = None
        original_step = env.step

        def observed_step(action):
            before = self.state.copy()
            result = original_step(action)
            if self.total_steps <= 4000:
                index = self.total_steps - 1
                errors = {'maximum_state_error': np.max(np.abs(before - self.original['state'][index])),
                          'maximum_action_error': np.max(np.abs(np.asarray(action).reshape(3) - self.original['action'][index])),
                          'maximum_reward_error': abs(result[1] - self.original['reward'][index]),
                          'maximum_position_error': np.max(np.abs(env.pos[0] - self.original['position'][index]))}
                self.prefix_errors['steps_checked'] += 1
                for k, value in errors.items():
                    self.prefix_errors[k] = max(self.prefix_errors[k], float(value))
                if any(value != 0. for value in errors.values()):
                    dump(self.root / 'prefix_failure.json', {'step': self.total_steps, 'errors': {k: float(v) for k, v in errors.items()}})
                    raise RuntimeError(f'Nonidentical prefix at upper step {self.total_steps}')
            return result
        env.step = observed_step

    def snapshot(self, label=None):
        step = self.total_steps
        with preserve_rng():
            self.observer.save(step)
            path = self.ckpts / f'model_{step:06d}_full_snapshot.pt'
            p = torch.load(path, map_location='cpu', weights_only=False)
            p['v2_branch'] = self.branch
            p['policy_anchor_weight'] = self.agent.policy_anchor_weight
            p['fixed_source_anchor'] = (self.agent._policy_anchor_actor.state_dict()
                                        if self.agent._policy_anchor_actor is not None else None)
            p['last_train_info'] = self.last_train_info
            p['snapshot_label'] = label
            # The 6k stop must retain both replay buffers, just like the 5k frame.
            if step == 6000 or label == 'raw_final':
                from scripts.reproduce_full_ft_v1_unfreeze import buffer_payload
                p['replay'] = buffer_payload(self.buffer)
                p['success_replay'] = {k: buffer_payload(v) for k, v in self.success_buffer.buffers.items()}
                p['replay_included'] = True
            torch.save(p, path)
        return p

    def verify_and_configure(self):
        actual = self.snapshot('verified_prefix_before_branch_changes')
        expected = torch.load(REFERENCE, map_location='cpu', weights_only=False)
        mismatches = [d for k in CHECK_FIELDS for d in differences(actual[k], expected[k], k)]
        result = {'reference': str(REFERENCE), 'reference_sha256': sha(REFERENCE),
                  'fields_checked': list(CHECK_FIELDS), 'mismatches': mismatches,
                  'all_fields_exactly_equal': not mismatches,
                  'trajectory': self.prefix_errors, 'torch_threads': torch.get_num_threads(),
                  'initial_networks_exactly_equal': not self.initial_differences,
                  'authorized_environment_seed_exception': 'existing seed43 prefix only'}
        dump(self.root / 'prefix_verification.json', result)
        if mismatches:
            raise RuntimeError(f'4k snapshot mismatch: {mismatches[:20]}')
        if torch.get_num_threads() != 6:
            raise RuntimeError('Deterministic prefix requires the verified six-thread CPU setting')
        actor_lr, anchor_weight = BRANCHES[self.branch]
        for group in self.agent.actor_optimizer.param_groups:
            group['lr'] = actor_lr
        for group in self.agent.critic_optimizer.param_groups:
            group['lr'] = 3e-4
        self.agent.policy_anchor_weight = anchor_weight
        self.agent.set_policy_anchor_from_current()
        anchor = self.agent._policy_anchor_actor
        if anchor_weight:
            assert anchor is not None and self.success_buffer.size > 0
            assert not differences(anchor.state_dict(), self.source_weights)
            assert all(not p.requires_grad for p in anchor.parameters())
        else:
            assert anchor is None

        def fixed_anchor_guard():
            if anchor_weight and (self.agent._policy_anchor_actor is not anchor
                                  or differences(anchor.state_dict(), self.source_weights)):
                raise RuntimeError('Fixed source anchor identity or parameters changed')
        self.agent.set_policy_anchor_from_current = fixed_anchor_guard
        self.best.on_train_end = self.select_final
        self.best.final_eval_callback = None
        # Additional diagnostics preserve the CPU RNG stream. Standard periodic
        # evaluations remain in their original position in the callback list.
        self.prefix_verified = True
        self.diagnose('baseline_4k')
        dump(self.root / 'configuration.json', {
            'actor_lr': actor_lr, 'critic_lr': 3e-4, 'policy_anchor_weight': anchor_weight,
            'anchor_states': 'current successful Full replay minibatch states',
            'source_anchor_fixed_until_end': True, 'args': vars(self.args),
            'extension_gate': {'full_stage0': .8, 'full_stage3': .7, 'k1_stage3': .8},
            'early_stop': 'two consecutive complete evaluations with all three success rates zero, earliest 6000',
        })
        print(f'[V2 prefix verified] branch={self.branch}, exact complete 4k state; actor_lr={actor_lr}, anchor={anchor_weight}', flush=True)

    def diagnose(self, label):
        with preserve_rng(), torch.no_grad():
            obs = self.bank['observations']
            actions = actor_batch(self.agent, obs)
            commands = command_metrics(self.source_actions, actions, self.bank['positions'], np.array([3.5, 0., 1.]))
            near = self.bank['distances'] < 1.2
            commands['near_goal'] = command_metrics(self.source_actions[near], actions[near], self.bank['positions'][near], np.array([3.5, 0., 1.]))
            states = torch.as_tensor(obs, dtype=torch.float32, device=self.agent.device)
            a = torch.as_tensor(actions, dtype=torch.float32, device=self.agent.device)
            src = torch.as_tensor(self.source_actions, dtype=torch.float32, device=self.agent.device)
            q1, q2 = self.agent.critic(states, a)
            qs1, qs2 = self.agent.critic(states, src)
            row = {'step': self.total_steps, 'actor_updates': self.observer.actor_updates,
                   'label': label, 'commands': commands,
                   'q': {'current_q1': float(q1.mean()), 'current_q2': float(q2.mean()),
                         'current_minq': float(torch.minimum(q1, q2).mean()),
                         'source_q1': float(qs1.mean()), 'source_q2': float(qs2.mean()),
                         'maximum_abs_q': float(torch.stack([q1, q2, qs1, qs2]).abs().max())},
                   'finite': bool(torch.isfinite(torch.stack([q1, q2, qs1, qs2])).all()
                                  and np.isfinite(actions).all()
                                  and all(torch.isfinite(p).all() for module in (self.agent.actor, self.agent.critic)
                                          for p in module.parameters())),
                   'last_train_info': self.last_train_info}
            np.savez_compressed(self.root / f'probe_{self.total_steps:06d}.npz',
                                actions=actions, source_actions=self.source_actions,
                                q1=q1.cpu().numpy(), q2=q2.cpu().numpy(),
                                source_q1=qs1.cpu().numpy(), source_q2=qs2.cpu().numpy())
        self.diagnostics.append(row)
        dump(self.root / 'diagnostics.json', self.diagnostics)
        print(f'[V2 probe] {self.branch} step={self.total_steps} actor_updates={self.observer.actor_updates} '
              f'drift={commands["mean_command_difference_mps"]:.4f} near_radial={commands["near_goal"]["updated_mean_goal_closing_command_mps"]:.4f}', flush=True)
        if not row['finite']:
            self.request_stop = True
            self.stop_reason = 'nonfinite_network_or_q_diagnostic'
        return row

    def read_evaluation(self):
        row = {'step': self.total_steps}
        for name in ('full_stage0', 'full_stage3', 'k1_stage3'):
            path = Path(self.args.log_dir) / f'{name}_validation' / 'evaluation_metrics.jsonl'
            records = [json.loads(line) for line in path.read_text().splitlines()]
            matches = [r for r in records if r['step'] == self.total_steps]
            if len(matches) != 1:
                raise RuntimeError(f'Evaluation not unique at {path}: {self.total_steps}')
            row[name] = matches[0]
            assert [ep['environment_seed'] for ep in row[name]['episodes']] == list(range(80, 90))
        self.evaluations.append(row)
        dump(self.root / 'evaluations.json', self.evaluations)
        return row

    def on_step(self, trainer):
        info = self.last_train_info or {}
        if info.get('actor_updated'):
            self.observer.actor_updates += 1
            self.anchor_loss_updates.append({'step': self.total_steps, 'actor_update': self.observer.actor_updates,
                                            **info})
            if self.agent.policy_anchor_weight:
                self.agent.set_policy_anchor_from_current()
                assert info['success_replay_fraction'] > 0
        if self.total_steps == 4000:
            self.verify_and_configure()
            return
        if not self.prefix_verified:
            return
        probe_update = info.get('actor_updated') and self.observer.actor_updates in UPDATE_PROBES
        evaluate = self.total_steps % 5000 == 0 or self.total_steps == 6000
        if probe_update or evaluate:
            self.snapshot('actor_update_probe' if probe_update else 'stability_evaluation')
            self.diagnose('actor_update_probe' if probe_update else 'stability_evaluation')
            dump(self.root / 'actor_updates.json', self.anchor_loss_updates)
        if evaluate:
            if self.total_steps == 6000:
                with preserve_rng():
                    self.triple.on_step(self, force=True)
                self.best.on_step(self)
            row = self.read_evaluation()
            post = [r for r in self.evaluations if r['step'] >= 5000]
            if len(post) >= 2 and fully_collapsed(post[-1]) and fully_collapsed(post[-2]):
                self.request_stop = True
                self.stop_reason = 'persistent_full_collapse_all_three_domains'
                self.stop_decision = {'step': self.total_steps, 'reason': self.stop_reason,
                                      'evaluations': [post[-2]['step'], post[-1]['step']]}
                dump(self.root / 'stop_decision.json', self.stop_decision)
            if self.total_steps == 10000:
                by_step = {r['step']: r for r in self.evaluations}
                extend = (stable_for_extension(by_step[5000]) and stable_for_extension(row)
                          and all(d['finite'] for d in self.diagnostics))
                self.extension_decision = {'step': 10000, 'extend_to_40000': extend,
                                           'criteria': {'full_stage0': .8, 'full_stage3': .7, 'k1_stage3': .8},
                                           'evaluations': [5000, 10000]}
                dump(self.root / 'extension_decision.json', self.extension_decision)
                if not extend:
                    self.request_stop = True
                    self.stop_reason = self.stop_reason or '10k_extension_stability_gate_not_met'
        if self.request_stop:
            self.snapshot('raw_final')
            dump(self.root / 'actor_updates.json', self.anchor_loss_updates)
            self.raw_end_diagnostics = self.diagnostics[-1]
            self.raw_stop_reason = self.stop_reason

    def on_episode_end(self, trainer):
        if self.total_steps > 4000:
            info = self.last_info or {}
            self.training_episodes.append({'step': self.total_steps, 'episode': self._full_ft_episode_index,
                                           'stage': getattr(self, '_full_ft_trace_stage', None),
                                           'success': bool(info.get('success', False)), 'done_reason': info.get('done_reason')})
            dump(self.root / 'training_episodes.json', self.training_episodes)

    def select_final(self, trainer):
        qualified = self.best.best_step is not None
        if not qualified:
            # Preserve a named best file, while explicitly declaring that no
            # post-unfreeze model passed the existing stable-window selector.
            for suffix in ('actor', 'critic', 'actor_optimizer', 'critic_optimizer'):
                source = self.ckpts / f'model_004000_{suffix}'
                (self.ckpts / f'model_best_{suffix}').write_bytes(source.read_bytes())
            dump(self.ckpts / 'model_best_metrics.json', {
                'step': 4000, 'selection': 'unchanged_source_actor_fallback',
                'qualified_finetuned_best': False, 'reason': 'no qualified stable post-unfreeze checkpoint'})
        else:
            path = self.ckpts / 'model_best_metrics.json'
            value = json.loads(path.read_text()); value['qualified_finetuned_best'] = True
            dump(path, value)
        self.agent.load(str(self.ckpts / 'model_best'))
        dump(self.ckpts / 'model_final_selection.json', {
            'selected_checkpoint': 'model_best',
            'selected_checkpoint_step': self.best.best_step if qualified else 4000,
            'qualified_finetuned_best': qualified, 'model_final_is_restored_selection': True,
            'raw_final_step': self.total_steps,
            'raw_evaluation_file': str(self.root / 'evaluations.json'),
            'selected_actor_sha256': sha(self.ckpts / 'model_best_actor'),
            'raw_actor_sha256': sha(self.ckpts / 'model_raw_final_actor'),
        })

    def on_train_end(self, trainer):
        # This observer is moved last, after raw saving and best selection.
        dump(self.root / 'result.json', {
            'branch': self.branch, 'completed': True, 'prefix_verified': self.prefix_verified,
            'upper_steps': self.total_steps, 'actor_updates': self.observer.actor_updates,
            'stop_reason': getattr(self, 'raw_stop_reason', '40000_budget_complete'),
            'extension_decision': self.extension_decision,
            'early_stop': self.stop_decision,
            'model_raw_final': str(self.ckpts / 'model_raw_final'),
            'model_best': str(self.ckpts / 'model_best'),
            'best_selection': json.loads((self.ckpts / 'model_best_metrics.json').read_text()),
            'raw_final_diagnostics': self.diagnostics[-1],
            'baseline': self.evaluations[0], 'raw_final_evaluation': self.evaluations[-1],
        })

    def run(self):
        self.triple = next(cb for cb in self.callbacks if isinstance(cb, FullFTTripleEvalCallback))
        self.best = next(cb for cb in self.callbacks if isinstance(cb, BestCheckpointCallback))
        # Diagnostics and the 6k extra evaluation must run after ordinary eval,
        # best selection and checkpoint saving, and before end-of-run delivery.
        self.callbacks.remove(self)
        self.callbacks.append(self)
        baseline = self.read_evaluation()
        self.snapshot('initial_before_prefix')
        try:
            super().run()
        except BaseException as exc:
            dump(self.root / 'failure.json', {'error_type': type(exc).__name__, 'error': str(exc),
                                            'step': self.total_steps, 'prefix_verified': self.prefix_verified})
            raise


def run_branch(branch):
    root = ROOT / branch
    if (root / 'result.json').exists():
        print(f'[V2 completed] {branch}: reading existing result; no rerun')
        return
    if (root / 'prefix_verification.json').exists() or (CKPTS / branch / 'model_initial_actor').exists():
        raise RuntimeError(f'Incomplete branch {branch} already exists; refusing to overwrite its evidence')
    root.mkdir(parents=True, exist_ok=True)
    source = Path('checkpoints/startobscurr_v1_selected_20261010/seed43/model_final_actor')
    manifest = json.loads((source.parent.parent / 'manifest.json').read_text())
    assert sha(source) == manifest['seeds']['43']['files']['model_final_actor']['sha256']
    torch.set_num_threads(6)
    AblationTrainer.branch = branch
    command = build_mode_cmd('full_ft_v1')[3:]
    replacements = {'--seed': 43, '--total_steps': 40000,
                    '--full_ft_actor_source': str(source).removesuffix('_actor'),
                    '--log_dir': f'runs/full_ft_v2_seed43/{branch}', '--ckpt_dir': CKPTS / branch,
                    '--full_ft_trace_dir': root / 'training_traces',
                    '--trajectory_output_dir': root / 'trajectory',
                    '--collision_diagnostic_output_dir': root / 'collisions',
                    '--timeout_diagnostic_output_dir': root / 'timeouts',
                    '--eval_terminal_coverage_output_dir': root / 'coverage'}
    for option, value in replacements.items():
        command[command.index(option) + 1] = str(value)
    with patch.object(change_td3, 'TD3Trainer', AblationTrainer), patch.object(sys, 'argv', [__file__] + command):
        change_td3.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch', choices=tuple(BRANCHES))
    parser.add_argument('--suite', action='store_true')
    args = parser.parse_args()
    if args.suite == (args.branch is not None):
        parser.error('Choose exactly one of --suite or --branch')
    if args.branch:
        run_branch(args.branch)
    else:
        # Separate processes retain the original initialization and RNG setup.
        ROOT.mkdir(parents=True, exist_ok=True)
        manifest_path = ROOT / 'original_artifact_hashes_before.json'
        if manifest_path.exists():
            before = json.loads(manifest_path.read_text())
        else:
            before = protected_hashes()
            dump(manifest_path, before)
        for branch in BRANCHES:
            ROOT.mkdir(parents=True, exist_ok=True)
            with (ROOT / f'{branch}.log').open('a') as log:
                result = subprocess.run([sys.executable, '-u', '-m', 'scripts.run_full_ft_v2', '--branch', branch],
                                        stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                dump(ROOT / 'suite_failure.json', {'branch': branch, 'returncode': result.returncode,
                                                  'remaining_branches_not_started': list(BRANCHES)[list(BRANCHES).index(branch) + 1:]})
                raise SystemExit(result.returncode)
        after = protected_hashes()
        changed = [p for p in before if after.get(p) != before[p]]
        dump(ROOT / 'original_artifact_integrity.json', {
            'protected_files': len(before), 'changed_or_missing_files': changed,
            'all_original_files_unchanged': not changed})
        if changed:
            raise RuntimeError(f'Original artifact integrity failed: {changed[:10]}')
        from scripts.summarize_full_ft_v2 import summarize
        summarize()


if __name__ == '__main__':
    main()
