"""Read-only audit of the observed P0 vertical saturation; no learner edits."""
import json
from pathlib import Path

import numpy as np
import torch

from trainers.full_scratch import fingerprint, world_goal_delta, policy_diagnostics


def main():
    root = Path('checkpoints/full_scratch_ego20x10_curr_v1_seed43')
    audit = Path('runs/full_scratch_ego20x10_curr_v1_seed43/p0_audit')
    saved = torch.load(root / 'training_state_step6000.pt', map_location='cpu', weights_only=False)
    old = torch.load(root / 'training_state_step2000.pt', map_location='cpu', weights_only=False)
    agent = saved['agent']
    replay = saved['replay']['arrays']
    rows = [json.loads(r) for r in (audit / 'transitions.jsonl').read_text().splitlines()]
    episodes = [json.loads(r) for r in (audit / 'episodes.jsonl').read_text().splitlines()]
    updates = [json.loads(r) for r in (audit / 'updates.jsonl').read_text().splitlines()]
    for index, row in enumerate(rows):
        assert replay['not_done'][index, 0] == 1 - float(row['terminated'] or row['truncated'])
        np.testing.assert_allclose(replay['reward'][index, 0], row['replay_reward'], rtol=1e-6)
        assert row['lower_ticks'] == 8 or row['terminated'] or row['truncated']
        assert row['lower_ticks'] in range(1, 9)
    assert int(np.sum(replay['not_done'] == 0)) == len(episodes)
    assert tuple(saved['success']['strata']) == (10, 20, 30, 40)
    assert saved['success']['strata'][10]['size'] > 0
    assert all(r['success_replay/level_10_sample_count'] == 51 for r in updates)
    assert all(r['actor_frozen'] == r['conservative_phase'] == 0 and r['policy_freq'] == 2 and r['actor_lr'] == r['critic_lr'] == .0003 for r in updates)
    assert agent['tau'] == .005 and agent['discount'] == .99
    assert fingerprint(agent['actor_target']) != fingerprint(agent['actor'])
    assert fingerprint(agent['critic_target']) != fingerprint(agent['critic'])
    # Separate near and far probes on actual replay states; no random sampling.
    states = replay['state']
    distances = np.linalg.norm(world_goal_delta(states, [8, 0, 1]), axis=1)
    groups = {}
    for label, mask in (('near', distances < 1.2), ('far', distances >= 1.2)):
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        indices = indices[np.linspace(0, len(indices)-1, min(256, len(indices)), dtype=int)]
        batch = torch.as_tensor(states[indices])
        with torch.no_grad():
            policy = agent['actor'](batch)
            initial_policy = old['agent']['actor'](batch)
        scanned = []
        for z in (-1., -.5, 0., .5, 1.):
            candidate = policy.clone()
            candidate[:, 2] = z
            with torch.no_grad():
                q1, q2 = agent['critic'](batch, candidate)
            scanned.append(dict(z=z, q1_mean=float(q1.mean()), q2_mean=float(q2.mean()),
                                q_min_mean=float(torch.minimum(q1, q2).mean())))
        actions = policy.detach().clone().requires_grad_()
        q1 = agent['critic'].Q1(batch, actions)
        gradient = torch.autograd.grad(q1.sum(), actions)[0][:, 2].detach().numpy()
        groups[label] = dict(count=len(indices), initial_vertical_action_mean=float(initial_policy[:, 2].mean()),
                            current_vertical_action_mean=float(policy[:, 2].mean()),
                            current_vertical_saturation_fraction=float((policy[:, 2].abs() >= .999).float().mean()),
                            q1_dz_positive_fraction=float(np.mean(gradient > 0)), q1_dz_mean=float(gradient.mean()), vertical_scan=scanned)
    # Correct the monitoring-only near/far classification on all snapshots.
    corrected = []
    previous_policy = None
    with np.load(audit / 'fixed_policy_probe.npz') as fixed:
        fixed_states, fixed_actions = fixed['states'], fixed['actions']
    from algos.td3.td3_upper_recurrent import TD3UpperRecurrent
    for step in (2000, 5000, 6000):
        snapshot = torch.load(root / f'training_state_step{step}.pt', map_location='cpu', weights_only=False)
        probe_agent = TD3UpperRecurrent.__new__(TD3UpperRecurrent)
        probe_agent.__dict__.update(snapshot['agent'])
        values, policy = policy_diagnostics(probe_agent, fixed_states, fixed_actions, [8, 0, 1])
        values['adjacent_policy_rms_change'] = None if previous_policy is None else float(np.sqrt(np.mean((policy-previous_policy)**2)))
        previous_policy = policy
        buf = snapshot['replay']
        ids = (buf['ptr']-1-np.arange(min(256, buf['size']))) % buf['max_size']
        visited = buf['arrays']['state'][ids]
        distances_m = np.linalg.norm(world_goal_delta(visited, [8, 0, 1]), axis=1)
        for label, mask in (('near', distances_m < 1.2), ('far', distances_m >= 1.2)):
            values[label + '/probe_count'] = int(mask.sum())
            if mask.any():
                subset, _ = policy_diagnostics(probe_agent, visited[mask], buf['arrays']['action'][ids][mask], [8, 0, 1])
                values.update({label + '/' + key: value for key, value in subset.items()})
        corrected.append({'step': step, **values})
    (audit / 'policy_diagnostics_world_corrected.json').write_text(json.dumps(corrected, indent=2) + '\n')
    terminals = []
    for row in episodes:
        if row['done_reason'] == 'success':
            terminals.append(dict(step=row['step'], length=row['length'], initial_distance=row['initial_goal_distance'],
                                  final_distance=row['goal_distance'], terminal_reward=row['replay_reward'],
                                  reward_terms=row['reward_terms']))
    report = dict(engineering_checks='PASS', observed_issue='Actor positive-z saturation and bound failures after online updates',
                  terminal_mask_count=len(episodes), monitoring_correction='Use inverted KIN positions for metric metres; original normalized/clipped goal features misclassified near/far. Training inputs and rewards unchanged.', corrected_policy_probes=corrected, successful_training_episodes=terminals,
                  success_replay_samples_per_update=51, actor_updates=sum(r['actor_updated'] for r in updates),
                  critic_updates=len(updates), probe_groups=groups,
                  max_preclip_critic_grad_norm=max(r['critic_grad_norm'] for r in updates),
                  max_preclip_actor_grad_norm=max(r['actor_grad_norm'] for r in updates),
                  gradient_clip=agent['grad_clip_norm'],
                  interpretation='Observed learned critic action preference on replay states; not proof of the ultimate root cause. Reset, tick reward aggregation, executed-action replay, masks, success strata, optimizer cadence and target lag checks pass. Do not extend training unconditionally or alter reward/GRU before a controlled audit.',
                  next_audit='Use the same saved replay minibatches to compare learned vertical Q/action gradients with short isolated Full rollouts and reconstructed TD targets. Keep probes outside production replay.')
    Path('analysis_outputs/full_scratch_ego20x10_curr_v1_p0/stability_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
