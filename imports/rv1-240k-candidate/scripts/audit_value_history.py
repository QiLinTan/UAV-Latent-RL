"""Frozen-model action/replay/history audit and multi-step value diagnostics.

No optimizer updates. Forced-prefix returns are compared to prefix reward plus
bootstrap at the switch back to the actor, never directly to Q(s, first action).
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

from data.replay_buffer import ReplayBuffer
from envs.preprocess import preprocess_state
from scripts import change_td3
from scripts.audit_spatial_terminal_consistency import planned_path
from scripts.diagnose_terminal_reward_rollouts import feasible, preset_args
from scripts.run_change_td3 import build_mode_cmd
from trainers.td3_trainer import TD3Trainer

PRESET = 'upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1'


def qvalues(agent, state, action):
    q1, q2 = agent.evaluate_critic_actions(state, np.asarray(action).reshape(1, 3))
    return [float(q1[0]), float(q2[0]), float(min(q1[0], q2[0]))]


def history_check(state, next_state, action):
    old, new = state[117:].reshape(8, 100), next_state[117:].reshape(8, 100)
    np.testing.assert_array_equal(old[1:], new[:-1])
    np.testing.assert_allclose(new[-1, 96:99], action, atol=2e-7)
    np.testing.assert_array_equal(next_state[15:111], new[-1, :96])


def replay_audit(env, agent, args):
    config = copy.copy(args)
    config.buffer_size = 2048
    config.success_replay_size = 2048
    runner = TD3Trainer(env, agent, config)
    captured = []
    original_step = env.step

    def capture(action):
        before = runner.state.copy()
        passed = np.asarray(action).reshape(3).copy()
        command = env._velocity_command(passed)
        velocity = env.vel[0].copy()
        obs, reward, term, trunc, info = original_step(action)
        nxt = preprocess_state(obs.reshape(-1))
        np.testing.assert_array_equal(np.asarray(action).reshape(3), passed)
        np.testing.assert_allclose(env.vel[0], env._next_kinematic_velocity(velocity, command), atol=2e-6)
        history_check(before, nxt, passed)
        captured.append((before, passed, nxt, reward*args.reward_scale, term or trunc))
        return obs, reward, term, trunc, info

    episodes = 0
    with patch.object(env, 'step', capture):
        # Actual trainer collection paths: warmup, noisy actor, deterministic actor.
        for mode in ('warmup', 'noisy', 'actor'):
            if mode == 'actor':
                obs, _ = env.reset(seed=0)
                runner.state = preprocess_state(obs.reshape(-1))
                runner._episode_success_transitions.clear()
                runner.episode_step = 0
            runner.total_steps = 0 if mode == 'warmup' else args.start_timesteps+1
            with patch.object(runner.noise_scheduler, 'get_noise', return_value=.2 if mode == 'noisy' else 0.):
                for _ in range(190 if mode == 'actor' else 24):
                    index = runner.buffer.ptr
                    runner.step_env()
                    before, action, nxt, reward, done = captured[-1]
                    np.testing.assert_array_equal(runner.buffer.state[index], before)
                    np.testing.assert_array_equal(runner.buffer.action[index], action)
                    np.testing.assert_array_equal(runner.buffer.next_state[index], nxt)
                    np.testing.assert_allclose(runner.buffer.reward[index, 0], reward, atol=2e-6)
                    assert runner.buffer.not_done[index, 0] == 1.-float(done)
                    if done:
                        episodes += 1
                        reset_history = runner.state[117:].reshape(8, 100)
                        np.testing.assert_array_equal(reset_history, np.tile(reset_history[0], (8, 1)))
                        np.testing.assert_array_equal(reset_history[:, 96:99], 0.)
                        if mode == 'actor':
                            break
    # Reordered, repeated sampling must preserve each window and paired action.
    indices = np.array([len(captured)-1, 0, 10, 0, 23])
    with patch('numpy.random.randint', return_value=indices):
        sample = runner.buffer.sample(len(indices))
    for tensor, array in zip(sample, (runner.buffer.state, runner.buffer.action,
            runner.buffer.next_state, runner.buffer.reward, runner.buffer.not_done)):
        np.testing.assert_array_equal(tensor.numpy(), array[indices])
    batch = sample[0].to(agent.device)
    with torch.no_grad():
        online = np.stack([agent.select_action(row) for row in sample[0].numpy()])
        batched = agent.actor(batch).cpu().numpy()
    np.testing.assert_allclose(online, batched, atol=2e-5)
    successes = runner.success_buffer.size
    for buffer in runner.success_buffer.buffers.values():
        for i in range(buffer.size):
            history_check(buffer.state[i], buffer.next_state[i], buffer.action[i])
    # Check production target-noise projection using the same helper as train().
    from algos.td3.td3_upper_recurrent import _project_planar_action_tensor
    with torch.no_grad():
        nxt = sample[2].to(agent.device)
        raw = (agent.actor_target(nxt)+torch.tensor([.5, .5, -.5], device=agent.device)).clamp(-1, 1)
        target = _project_planar_action_tensor(raw, 1.).cpu().numpy()
    for row, expected in zip(raw.cpu().numpy(), target):
        np.testing.assert_allclose(agent.project_action(row), expected, atol=2e-7)
    return dict(transitions=len(captured), reset_checks=episodes, success_replay_transitions=successes,
                online_batch_actor_max_error=float(np.max(abs(online-batched))), all_checks_passed=True)


def run(callback, trainer, args, cli):
    out = cli.output_dir
    out.mkdir(parents=True, exist_ok=False)
    env = callback.env_cls(**callback.env_kwargs, gui=False)
    agent = trainer.agent
    result = dict(checkpoint=cli.checkpoint, gamma=args.gamma, reward_scale=args.reward_scale,
                  actor_sha256=hashlib.sha256(Path(cli.checkpoint+'_actor').read_bytes()).hexdigest())
    rows = []
    try:
        result['replay_audit'] = replay_audit(env, agent, args)
        # Same spatial observation, different time-to-terminal: a finite-horizon probe.
        obs, _ = env.reset(seed=0)
        hold_states, hold_rewards = [], []
        for t in range(200):
            hold_states.append(preprocess_state(obs.reshape(-1)))
            obs, reward, term, trunc, info = env.step(np.zeros(3))
            hold_rewards.append(reward*args.reward_scale)
            if term or trunc:
                break
        result['clock_probe'] = dict(outcome=info['done_reason'], steps=len(hold_rewards),
            observation_max_difference_0_vs_150=float(np.max(abs(hold_states[0]-hold_states[150]))),
            zero_action_q=qvalues(agent, hold_states[0], np.zeros(3)),
            hold_return_0=float(sum(args.gamma**k*r for k,r in enumerate(hold_rewards))),
            hold_return_150=float(sum(args.gamma**k*r for k,r in enumerate(hold_rewards[150:]))),
            caveat='Holding policy probe demonstrates hidden horizon; these are not actor-policy Q targets.')
        for seed in (0, 3, 17):
            obs, _ = env.reset(seed=seed)
            original_actions, states = [], []
            for t in range(200):
                states.append(preprocess_state(obs.reshape(-1)))
                action = agent.select_action(states[-1])
                original_actions.append(action)
                obs, _, term, trunc, info = env.step(action)
                if term or trunc:
                    break
            for start in (0, 45, 90):
                if start >= len(states):
                    continue
                for behavior, horizon in (('actor', 0), ('planned', 15), ('planned', 45),
                                          ('brake', 15), ('planned_full', 200)):
                    obs, _ = env.reset(seed=seed)
                    for action in original_actions[:start]:
                        obs, _, term, trunc, _ = env.step(action)
                        assert not (term or trunc)
                    np.testing.assert_allclose(preprocess_state(obs.reshape(-1)), states[start], atol=1e-5)
                    path = planned_path(env) if behavior.startswith('planned') else None
                    if path is None and behavior.startswith('planned'):
                        result.setdefault('skipped_planner_branches', []).append(dict(seed=seed, start=start, behavior=behavior, horizon=horizon))
                        continue
                    waypoint = 0
                    rewards, trace = [], []
                    initial_distance = float(np.linalg.norm(env.TARGET_POS-env.pos[0]))
                    for k in range(200):
                        state = preprocess_state(obs.reshape(-1))
                        actor = agent.select_action(state)
                        if k < horizon and behavior.startswith('planned'):
                            while waypoint < len(path)-1 and np.linalg.norm(path[waypoint]-env.pos[0][:2]) < .12:
                                waypoint += 1
                            command = feasible(np.r_[2.5*(path[waypoint]-env.pos[0][:2]),
                                                       1.5*(env.TARGET_POS[2]-env.pos[0][2])], env)
                            action = command / [env.max_reference_speed, env.max_reference_speed, env.max_reference_vertical_speed]
                        elif k < horizon and behavior == 'brake':
                            action = np.zeros(3)
                        else:
                            action = actor
                        values = qvalues(agent, state, action)
                        actor_values = qvalues(agent, state, actor)
                        obs, reward, term, trunc, info = env.step(action)
                        rewards.append(float(reward)*args.reward_scale)
                        trace.append(dict(q=values, actor_q=actor_values,
                            distance=float(info['goal_distance']), action=np.asarray(action).tolist()))
                        if term or trunc:
                            break
                    returns = np.zeros(len(rewards)+1)
                    for k in range(len(rewards)-1, -1, -1):
                        returns[k] = rewards[k]+args.gamma*returns[k+1]
                    # The bootstrap point is exactly where forced control ends.
                    cut = min(horizon, len(rewards))
                    prefix = sum(args.gamma**k*r for k,r in enumerate(rewards[:cut]))
                    bootstrap = trace[cut]['actor_q'] if cut<len(rewards) else [0., 0., 0.]
                    estimate = [prefix+args.gamma**cut*q for q in bootstrap]
                    actor_nstep = {}
                    if behavior == 'actor':
                        for h in (1, 5, 15, 45):
                            h = min(h, len(rewards))
                            early = sum(args.gamma**k*r for k,r in enumerate(rewards[:h]))
                            tail = trace[h]['actor_q'] if h<len(rewards) else [0.,0.,0.]
                            actor_nstep[str(h)] = [early+args.gamma**h*q for q in tail]
                    row = dict(seed=seed, start=start, behavior=behavior, forced_steps=cut,
                        outcome=info['done_reason'], initial_distance=initial_distance,
                        final_distance=float(info['goal_distance']), minimum_distance=min(x['distance'] for x in trace),
                        return_actual=float(returns[0]), prefix_plus_actor_bootstrap=estimate,
                        switch_actual_actor_return=float(returns[cut]), switch_actor_q=bootstrap,
                        actor_nstep_estimates=actor_nstep, rewards=rewards, trace=trace)
                    rows.append(row)
            print(f'Completed env seed {seed}: {len(rows)} branches', flush=True)
            (out/'branches.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    finally:
        env.close()
    result['branch_count'] = len(rows)
    (out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output-dir', required=True, type=Path)
    cli = p.parse_args()
    if cli.output_dir.exists():
        raise FileExistsError(cli.output_dir)
    args = preset_args(PRESET)
    class AuditCallback(change_td3.EvalCallback):
        def on_step(self, trainer, *, force=False):
            run(self, trainer, args, cli)
    command = build_mode_cmd(PRESET)[3:]+['--evaluate_only','true','--load_checkpoint',cli.checkpoint,
        '--log_dir',str(cli.output_dir.parent/(cli.output_dir.name+'_loader'))]
    with patch.object(change_td3, 'EvalCallback', AuditCallback), patch.object(sys, 'argv', [__file__]+command):
        change_td3.main()


if __name__ == '__main__':
    main()
