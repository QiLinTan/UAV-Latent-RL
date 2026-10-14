"""Isolated P0 causal/value audit. Production code and outputs stay read-only."""
import argparse
import copy
import hashlib
import json
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p
import torch
from scipy.stats import spearmanr

from algos.td3.td3_upper_recurrent import TD3UpperRecurrent, _project_planar_action_tensor
from envs.forest.ego_curriculum_env import EgoCurriculumFullForestEnv
from envs.preprocess import preprocess_state
from scripts.audit_stage0_feasibility import make_stage0_env
from trainers.full_scratch import restore_buffer, fingerprint, world_goal_delta
from trainers.td3_trainer import TD3Trainer

SOURCE = Path('checkpoints/full_scratch_ego20x10_curr_v1_seed43')
LOGS = Path('runs/full_scratch_ego20x10_curr_v1_seed43')
DEFAULT = Path('analysis_outputs/full_scratch_ego20x10_curr_v1_value_audit_seed43')
ZGRID = np.array([-1., -.5, 0., .5, 1.], dtype=np.float32)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def rows(path):
    return [json.loads(row) for row in path.read_text().splitlines()]


def load(step):
    return torch.load(SOURCE / f'training_state_step{step}.pt', map_location='cpu', weights_only=False)


def learner(payload):
    agent = TD3UpperRecurrent.__new__(TD3UpperRecurrent)
    agent.__dict__.update(payload['agent'])
    return agent


def env_for(payload, **pool_kwargs):
    args = SimpleNamespace(**payload['protocol'])
    args.map_id = 'ego_forest_20x10_seed43_v1'  # factory placeholder is replaced by the frozen pool
    def factory(**kwargs):
        for key in tuple(kwargs):
            if key.startswith('kinematic_'):
                kwargs.pop(key)
        # Legacy audit factory assumes K1 (8/120); production Full uses 1/120.
        # Match the existing production reward exactly, without changing its definition.
        kwargs["reward_model"].integration_dt = 1. / args.ctrl_freq
        return EgoCurriculumFullForestEnv(**kwargs, **pool_kwargs)
    return make_stage0_env(args, env_class=factory)


def rng_restore(payload):
    np.random.set_state(payload['numpy_rng'])
    random.setstate(payload['python_rng'])
    torch.set_rng_state(payload['torch_rng'])


def describe(array):
    array = np.asarray(array)
    return dict(mean=float(array.mean()), std=float(array.std()), min=float(array.min()),
                p05=float(np.quantile(array, .05)), p50=float(np.quantile(array, .5)),
                p95=float(np.quantile(array, .95)), max=float(array.max()))


def fixed_indices(states, per_group=32):
    delta = world_goal_delta(states, [8, 0, 1])
    distance = np.linalg.norm(delta, axis=1)
    height = -delta[:, 2]
    chosen = list(range(min(64, len(states))))
    for mask in (distance < 1.2, distance >= 4., height > .15, height < -.15):
        ids = np.flatnonzero(mask)
        if len(ids):
            chosen.extend(ids[np.linspace(0, len(ids)-1, min(per_group, len(ids)), dtype=int)].tolist())
    return np.array(sorted(set(chosen)))


def probe(agent, states, previous=None, gradients=True):
    batch = torch.as_tensor(states)
    with torch.no_grad():
        action = agent.actor(batch)
        q1, q2 = agent.critic(batch, action)
    a = action.numpy()
    delta = world_goal_delta(states, [8, 0, 1])
    radial = delta[:, :2] / np.maximum(np.linalg.norm(delta[:, :2], axis=1, keepdims=True), 1e-9)
    result = dict(action_mean=a.mean(axis=0).tolist(), action_std=a.std(axis=0).tolist(),
                  z_positive_fraction=float(np.mean(a[:, 2] > 0)),
                  z_saturation_fraction=float(np.mean(np.abs(a[:, 2]) >= .999)),
                  z_above_09_fraction=float(np.mean(a[:, 2] >= .9)),
                  xy_boundary_fraction=float(np.mean(np.linalg.norm(a[:, :2], axis=1) >= .999)),
                  radial_command_mps=float(np.mean(np.sum(a[:, :2]*radial, axis=1)) * .8),
                  vertical_command_mps=float(a[:, 2].mean()*.3),
                  q1=describe(q1.numpy()), q2=describe(q2.numpy()), q_min=describe(torch.minimum(q1, q2).numpy()),
                  adjacent_policy_rms=None if previous is None else float(np.sqrt(np.mean((a-previous)**2))))
    if gradients:
        candidate = action.detach().requires_grad_()
        values = agent.critic.Q1(batch, candidate)
        gradient = torch.autograd.grad(values.sum(), candidate)[0].detach().numpy()
        result.update(q1_action_gradient_mean=gradient.mean(axis=0).tolist(),
                      q1_z_gradient_positive_fraction=float(np.mean(gradient[:, 2] > 0)),
                      q1_z_gradient=describe(gradient[:, 2]))
        scans = []
        for z in ZGRID:
            candidate = action.clone()
            candidate[:, 2] = float(z)
            with torch.no_grad():
                v1, v2 = agent.critic(batch, candidate)
            scans.append(dict(z=float(z), world_vz_mps=float(z*.3), q1=describe(v1.numpy()),
                              q2=describe(v2.numpy()), q_min=describe(torch.minimum(v1, v2).numpy())))
        result['vertical_scan'] = scans
    return result, a


def reconstruct_5k(trainer, two, five, six):
    """Rebuild the *same* in-progress episode; do not reset at 5k."""
    env = trainer.env
    env.start_curriculum.__dict__.update(copy.deepcopy(two['start']))
    env.obstacle_curriculum.__dict__.update(copy.deepcopy(two['obstacle']))
    env._start_rng.bit_generator.state = copy.deepcopy(two['start_rng'])
    env.base_env._rng.bit_generator.state = copy.deepcopy(two['map_rng'])
    env.base_env._reset_count = two['reset_count']
    env.restore_sampler_state(two['pool_sampler'])
    observation, info = env.reset()  # original 2k->6k physical resume boundary
    trainer.state = preprocess_state(observation)
    trainer._reset_episode_coverage(info)
    trainer._episode_success_transitions.clear()
    trainer.episode_step = 0
    trainer.episode_return = 0.
    replay = six['replay']['arrays']
    max_error = 0.
    for step in range(2001, 5001):
        index = step-1
        error = float(np.max(np.abs(trainer.state-replay['state'][index])))
        max_error = max(max_error, error)
        np.testing.assert_allclose(trainer.state, replay['state'][index], atol=2e-6, rtol=0)
        action = replay['action'][index]
        observation, reward, terminated, truncated, info = env.step(action.reshape(1, 3))
        next_state = preprocess_state(observation)
        np.testing.assert_allclose(next_state, replay['next_state'][index], atol=2e-6, rtol=0)
        np.testing.assert_allclose(reward*trainer.args.reward_scale, replay['reward'][index, 0], rtol=1e-6, atol=1e-6)
        assert 1-float(terminated or truncated) == replay['not_done'][index, 0]
        trainer._episode_success_transitions.append((trainer.state.copy(), action.copy(), float(replay['reward'][index, 0]), next_state.copy(), bool(terminated or truncated)))
        trainer.state = next_state
        trainer._record_episode_coverage(info)
        trainer.episode_step += 1
        trainer.episode_return += reward
        if terminated or truncated:
            env.record_episode_outcome(info['success'], info['collision'], info['done_reason'])
            observation, reset_info = env.reset()
            trainer.state = preprocess_state(observation)
            trainer._reset_episode_coverage(reset_info)
            trainer._episode_success_transitions.clear()
            trainer.episode_step = 0
            trainer.episode_return = 0.
        if step % 1000 == 0:
            print(f'[Reconstruct] {step} obs_max_error={max_error:g}', flush=True)
    assert env.sampler_state() == five['pool_sampler']
    assert env.start_curriculum.level == five['start']['level']
    assert list(env.start_curriculum.successes) == list(five['start']['successes'])
    assert trainer.episode_step == five['interrupted_episode_steps']
    trainer.total_steps = 5000
    trainer.last_info = info
    trainer.last_action = action
    restore_buffer(trainer.buffer, five['replay'])
    restore_buffer(trainer.success_buffer, five['success'])
    rng_restore(five)
    return dict(upper_steps_replayed=3000, maximum_pre_observation_error=max_error,
                episode_step_at_5k=trainer.episode_step, pool_sampler_identical=True,
                learner_total_it=trainer.agent.total_it, partial_success_candidates=len(trainer._episode_success_transitions))


class UpdateAudit:
    def __init__(self, trainer, fixed, root):
        self.trainer, self.fixed, self.root = trainer, fixed, root
        self.sampled = {}
        self.actor_output_gradient = None
        self.handles = []
        self.handles.append(trainer.agent.actor.register_forward_hook(self.actor_hook))
        self.handles.append(trainer.agent.actor_target.register_forward_hook(self.target_actor_hook))
        self.handles.append(trainer.agent.critic_target.register_forward_hook(self.target_critic_hook))
        self.handles.append(trainer.agent.critic.register_forward_hook(self.critic_hook))
        self.wrap_buffer(trainer.buffer, 'main')
        self.wrap_buffer(trainer.success_buffer, 'success')

    def actor_hook(self, module, args, output):
        if output.requires_grad:
            self.in_probe = True
            try:
                self.fixed_before_actor, _ = probe(self.trainer.agent, self.fixed, gradients=True)
            finally:
                self.in_probe = False
            output.register_hook(lambda gradient: setattr(self, 'actor_output_gradient', gradient.detach().clone()))

    def target_actor_hook(self, module, args, output):
        self.target_actor = output.detach().clone()

    def target_critic_hook(self, module, args, output):
        self.target_action = args[1].detach().clone()
        self.target_qs = tuple(q.detach().clone() for q in output)

    def critic_hook(self, module, args, output):
        if getattr(self, "in_probe", False):
            return
        self.current_qs = tuple(q.detach().clone() for q in output)

    def wrap_buffer(self, buffer, name):
        original = buffer.sample
        def sampled(count):
            before = np.random.get_state()
            local = np.random.RandomState()
            local.set_state(before)
            if name == 'main':
                indices = local.randint(0, buffer.size, size=count)
            else:
                assert buffer.buffers[10].size > 0 and all(b.size == 0 for k, b in buffer.buffers.items() if k != 10)
                indices = local.randint(0, buffer.buffers[10].size, size=count)
                indices = indices[local.permutation(count)]
            batch = original(count)
            self.sampled[name] = dict(indices=indices, batch=tuple(t.clone() for t in batch))
            return batch
        buffer.sample = sampled

    def reset(self):
        self.sampled.clear()
        self.actor_output_gradient = None
        self.torch_before = torch.get_rng_state().clone()

    def validate(self, step, info):
        agent = self.trainer.agent
        combined = [torch.cat([self.sampled[name]['batch'][i] for name in ('main', 'success')]) for i in range(5)]
        state, action, next_state, reward, mask = combined
        generator = torch.Generator().set_state(self.torch_before)
        noise = (torch.randn(action.shape, generator=generator)*agent.policy_noise).clamp(-agent.noise_clip, agent.noise_clip)
        expected_action = (self.target_actor+noise).clamp(-agent.max_action, agent.max_action)
        expected_action = _project_planar_action_tensor(expected_action, agent.max_action)
        torch.testing.assert_close(expected_action, self.target_action, rtol=0, atol=0)
        target = reward + mask * agent.discount * torch.minimum(*self.target_qs)
        loss = ((self.current_qs[0]-target)**2).mean() + ((self.current_qs[1]-target)**2).mean()
        assert abs(float(loss)-info['critic_loss']) < 1e-6
        assert abs(float(target.mean())-info['q_target_mean']) < 1e-6
        for name, buffer in (('main', self.trainer.buffer), ('success', self.trainer.success_buffer.buffers[10])):
            sampled = self.sampled[name]
            for column, key in enumerate(('state', 'action', 'next_state', 'reward', 'not_done')):
                np.testing.assert_array_equal(sampled['batch'][column].numpy(), getattr(buffer, key)[sampled['indices']])
        record = dict(step=step, critic_it=agent.total_it, main_count=len(self.sampled['main']['indices']), success_count=len(self.sampled['success']['indices']),
                      terminal_count=int((mask==0).sum()), target_mean=float(target.mean()),
                      target=describe(target.numpy()), reward=describe(reward.numpy()),
                      noise=describe(noise.numpy()), target_action=describe(self.target_action.numpy()),
                      critic_loss_reconstructed=float(loss), reconstruction_absolute_error=abs(float(loss)-info['critic_loss']))
        if self.actor_output_gradient is not None:
            derivative = -self.actor_output_gradient.numpy() * len(state)
            record['fixed_before_actor_update'] = self.fixed_before_actor
            record['actor_batch_q1_action_gradient_mean'] = derivative.mean(axis=0).tolist()
            record['actor_batch_positive_z_gradient_fraction'] = float(np.mean(derivative[:, 2] > 0))
            record['actor_output_loss_gradient_z_mean'] = float(self.actor_output_gradient[:, 2].mean())
            record['head_bias_gradient'] = agent.actor.policy[-1].bias.grad.numpy().tolist()
        if step in (5001, 5020, 5100, 5200, 5500, 6000):
            np.savez_compressed(self.root / 'td_batches' / f'step{step}.npz',
                                state=state.numpy(), action=action.numpy(), next_state=next_state.numpy(), reward=reward.numpy(), not_done=mask.numpy(),
                                target_actor=self.target_actor.numpy(), noise=noise.numpy(), smoothed_target_action=self.target_action.numpy(),
                                target_q1=self.target_qs[0].numpy(), target_q2=self.target_qs[1].numpy(), td_target=target.numpy(),
                                main_indices=self.sampled['main']['indices'], success_indices=self.sampled['success']['indices'])
        return record


def timeline(root, final_step=6000):
    two, five, six = load(2000), load(5000), load(6000)
    shutil.copyfile(SOURCE / 'training_state_step5000.pt', root / 'source_5k_complete.pt')
    env = env_for(five)
    args = SimpleNamespace(**five['protocol'], total_steps=6000, resume_training=None,
                           log_dir=str(root), ckpt_dir=str(root / 'checkpoints'))
    agent = learner(five)
    trainer = TD3Trainer(env, agent, args)
    reconstruction = reconstruct_5k(trainer, two, five, six)
    write(root / 'reconstruction.json', reconstruction)
    fixed_ids = fixed_indices(five['replay']['arrays']['state'])
    fixed = five['replay']['arrays']['state'][fixed_ids].copy()
    np.savez_compressed(root / 'fixed_states.npz', states=fixed, replay_indices=fixed_ids)
    (root / 'checkpoints').mkdir()
    (root / 'td_batches').mkdir()
    instrumentation = UpdateAudit(trainer, fixed, root)
    trajectory, update_rows, probes = [], [], []
    previous = None
    def snapshot(step):
        nonlocal previous
        values, actions = probe(agent, fixed, previous)
        previous = actions.copy()
        values.update(step=step, critic_it=agent.total_it, actor_updates=agent.total_it//2,
                      actor_sha256=fingerprint(agent.actor), critic_sha256=fingerprint(agent.critic))
        probes.append(values)
        agent.save(str(root / 'checkpoints' / f'model_raw_{step}'))
        torch.save(dict(actor_target=agent.actor_target.state_dict(), critic_target=agent.critic_target.state_dict(),
                        total_it=agent.total_it), root / 'checkpoints' / f'targets_{step}.pt')
        np.savez_compressed(root / 'checkpoints' / f'fixed_actions_{step}.npz', actions=actions)
        write(root / 'timeline_probes.json', probes)
        print(f'[Probe] {step} actor_updates={agent.total_it//2} az={values["action_mean"][2]:.6f} sat={values["z_saturation_fraction"]:.3f}', flush=True)
    snapshot(5000)
    original = six['replay']['arrays']
    try:
        for step in range(5001, final_step+1):
            trainer.total_steps = step
            before = trainer.state.copy()
            deterministic = agent.select_action(before)
            map_info = env._pool_info()
            trainer.step_env()
            np.testing.assert_allclose(trainer.buffer.state[step-1], original['state'][step-1], atol=2e-6, rtol=0)
            np.testing.assert_array_equal(trainer.buffer.action[step-1], original['action'][step-1])
            trajectory.append(dict(step=step, deterministic_action=deterministic.tolist(), executed_action=trainer.last_action.tolist(),
                                   height=float(before[2]*.5+1), vertical_velocity=float(before[8]*2),
                                   goal_distance=float(np.linalg.norm(world_goal_delta(before.reshape(1,-1), [8,0,1]))),
                                   done_reason=trainer.last_info['done_reason'], **map_info))
            instrumentation.reset()
            info = trainer.train_step()
            record = instrumentation.validate(step, info)
            record.update(info)
            update_rows.append(record)
            # Fine timing on one fixed state set. Deterministic, no RNG use.
            light, _ = probe(agent, fixed, gradients=False)
            record['fixed_z_mean_after_update'] = light['action_mean'][2]
            record['fixed_z_saturation_after_update'] = light['z_saturation_fraction']
            if step % 100 == 0:
                snapshot(step)
        expected = six['agent']
        if final_step != 6000:
            for name in ('actor', 'critic'):
                expected[name].load_state_dict(torch.load(root.parent/'checkpoints'/f'model_raw_{final_step}_{name}', weights_only=True))
            targets = torch.load(root.parent/'checkpoints'/f'targets_{final_step}.pt',weights_only=True)
            for name in ('actor_target','critic_target'):
                expected[name].load_state_dict(targets[name])
        matches = {name: fingerprint(getattr(agent,name)) == fingerprint(expected[name]) for name in ('actor','critic','actor_target','critic_target')}
        assert all(matches.values()), matches
        for key in original:
            np.testing.assert_array_equal(getattr(trainer.buffer, key)[:final_step], original[key][:final_step])
        for key, saved in six['success']['strata'].items():
            for name, values in saved['arrays'].items():
                np.testing.assert_array_equal(getattr(trainer.success_buffer.buffers[key], name)[:saved['size']], values)
        write(root / 'replication.json', dict(network_hash_matches=matches, replay_exact=True, success_replay_exact=True,
                                             critic_it=agent.total_it, new_critic_updates=final_step-5000, new_actor_updates=(final_step-5000)//2,
                                             sampler_exact=env.sampler_state()==six['pool_sampler'] if final_step==6000 else None))
        write(root / 'executed_vs_actor.json', trajectory)
        write(root / 'update_audit.json', update_rows)
    finally:
        for handle in instrumentation.handles:
            handle.remove()
        env.close()


def relocate_reset(env, position):
    """Diagnostic reset placement, mirroring the existing start wrapper."""
    position = np.asarray(position, dtype=np.float32)
    assert not env._place_kinematic_state(position, np.zeros(3, dtype=np.float32))
    env._episode_start_position = np.asarray(env.pos[0], dtype=np.float32).copy()
    base = env.base_env
    base._prev_pos = env._episode_start_position.copy()
    base._prev_goal_dist = float(np.linalg.norm(base.TARGET_POS-base._prev_pos))
    base._last_collision = False
    base._last_clearance = base._computeNearestTreeClearance(base._prev_pos)
    base._last_done_reason = 'running'
    env.reference_position = env._episode_start_position.copy()
    env.reference_velocity.fill(0)
    env.reference_acceleration.fill(0)
    env.command_velocity.fill(0)
    observation = base._computeObs()
    env._semantic_history.clear()
    frame = env._semantic_frame(observation)
    for _ in range(env.semantic_history_length):
        env._semantic_history.append(frame.copy())
    return env._upper_observation(observation)


def reconstruct_state(env, index, replay, resets):
    reset = next(r for r in reversed(resets) if r['step'] <= index)
    env.set_curriculum_stage_override(reset['density_stage'])
    env.reset(seed=reset['map_pool_index'])
    observation = relocate_reset(env, reset['start'])
    env._start_source = reset['source']
    env._start_level = reset['level']
    for previous in range(reset['step'], index):
        np.testing.assert_allclose(preprocess_state(observation), replay['state'][previous], atol=2e-6, rtol=0)
        observation, reward, terminated, truncated, info = env.step(replay['action'][previous].reshape(1,3))
        assert not terminated and not truncated
        np.testing.assert_allclose(reward*.1, replay['reward'][previous,0], atol=1e-6, rtol=1e-6)
    error = float(np.max(np.abs(preprocess_state(observation)-replay['state'][index])))
    np.testing.assert_allclose(preprocess_state(observation), replay['state'][index], atol=2e-6, rtol=0)
    return observation, reset, error


def python_state(env):
    return dict(wrapper=copy.deepcopy({k:v for k,v in env.__dict__.items() if k not in ('base_env','pool')}),
                base=copy.deepcopy(env.base_env.__dict__))


def restore_physics(env, state_id, state):
    p.restoreState(stateId=state_id, physicsClientId=env.base_env.CLIENT)
    env.base_env.__dict__.update(copy.deepcopy(state['base']))
    env.__dict__.update(copy.deepcopy(state['wrapper']))


def stable_suffix(env):
    """QA waypoint/height feedback only for an isolated physical diagnostic."""
    position, velocity = env.pos[0], env.vel[0]
    path = env.pool.path_for(env.pool_split, env._pool_index, env._density_episode_stage)
    nearest = int(np.argmin(np.linalg.norm(path[:,:2]-position[:2],axis=1)))
    waypoint = min(len(path)-1, nearest+4)
    delta = path[waypoint]-position
    if np.linalg.norm(position-env.TARGET_POS) < .8:
        delta = env.TARGET_POS-position
    command = 1.2*delta - .6*velocity
    action = command / np.array([.8,.8,.3])
    action[:2] *= min(1., 1./max(np.linalg.norm(action[:2]),1e-9))
    return np.clip(action,-1,1).astype(np.float32)


def rollout_intervention(env, observation, first_action, actor, critic, suffix, horizon=90):
    cumulative = 0.
    returns = {}
    minimum = float(np.linalg.norm(env.pos[0]-env.TARGET_POS))
    samples = []
    terminated = truncated = False
    for tick in range(horizon):
        if tick == 0:
            action = first_action
        elif suffix == 'frozen_actor_6k':
            with torch.no_grad():
                action = actor(torch.as_tensor(preprocess_state(observation)).reshape(1,-1))[0].numpy()
        else:
            action = stable_suffix(env)
        observation, reward, terminated, truncated, info = env.step(action.reshape(1,3))
        cumulative += .1*reward * .99**tick
        minimum = min(minimum, info['goal_distance'])
        samples.append(dict(action_z=float(action[2]), z=float(env.pos[0,2]), vz=float(env.vel[0,2]),
                            tracking_error=float(info['reference_tracking_error_mean']), reward=float(reward)))
        if tick+1 in (1,30,90) or terminated or truncated:
            with torch.no_grad():
                state = torch.as_tensor(preprocess_state(observation)).reshape(1,-1)
                tail_action = actor(state)
                v1,v2 = critic(state, tail_action)
            bootstrap = 0. if terminated or truncated else .99**(tick+1)*float(torch.minimum(v1,v2))
            returns[str(tick+1)] = dict(discounted_scaled_return=cumulative, bootstrapped_q_min=cumulative+bootstrap,
                                       bootstrap_tail=bootstrap, terminal=bool(terminated or truncated))
        if terminated or truncated:
            break
    for limit in (30,90):
        if str(limit) not in returns:
            returns[str(limit)] = copy.deepcopy(returns[str(tick+1)])
    return dict(returns=returns, actual_upper_steps=tick+1, minimum_distance=minimum,
                final_distance=float(info['goal_distance']), final_height=float(env.pos[0,2]),
                done_reason=info['done_reason'] if terminated or truncated else 'diagnostic_horizon', samples=samples)


def values(root):
    saved = load(6000)
    five = load(5000)
    replay = saved['replay']['arrays']
    states = replay['state']
    delta = world_goal_delta(states, [8,0,1])
    distance = np.linalg.norm(delta,axis=1)
    height = -delta[:,2]
    groups = {'near': distance < 1.2, 'far': distance >= 4.,
              'above': height > .15, 'below': height < -.15}
    candidates = {}
    for label, mask in groups.items():
        ids = np.flatnonzero(mask)
        selected = ids[np.linspace(0,len(ids)-1,min(3,len(ids)),dtype=int)]
        for index in selected:
            candidates.setdefault(int(index), []).append(label)
    write(root/'representative_indices.json', candidates)
    env = env_for(saved, training_course=False, pool_split='train')
    actors = {5000:five['agent']['actor'],6000:saved['agent']['actor']}
    critics = {5000:five['agent']['critic'],6000:saved['agent']['critic']}
    resets = rows(LOGS/'p0_audit/resets.jsonl')
    result = []
    try:
        for index, labels in candidates.items():
            observation, reset, error = reconstruct_state(env,index,replay,resets)
            tensor = torch.as_tensor(preprocess_state(observation)).reshape(1,-1)
            with torch.no_grad():
                policy = actors[6000](tensor)[0].numpy()
                actor_values = {str(step): [float(q) for q in critics[step](tensor,actors[step](tensor))] for step in actors}
            scans = []
            for vertical in ZGRID:
                action = policy.copy()
                action[2] = vertical
                with torch.no_grad():
                    qs = {str(step): [float(q) for q in critics[step](tensor,torch.as_tensor(action).reshape(1,3))] for step in critics}
                for suffix in ('frozen_actor_6k','waypoint_height_feedback'):
                    rebuilt_observation, _, rebuilt_error = reconstruct_state(env,index,replay,resets)
                    np.testing.assert_array_equal(rebuilt_observation, observation)
                    assert rebuilt_error == error
                    physical = rollout_intervention(env,observation.copy(),action,actors[6000],critics[6000],suffix)
                    scans.append(dict(normalized_z=float(vertical), world_vz_mps=float(vertical*.3), xy_action=policy[:2].tolist(),
                                      q=qs, suffix=suffix, **physical))
            # Rebuild from reset and every recorded action: solver caches and
            # PID/reference/history evolve exactly as in the real trajectory.
            reconstruct_state(env,index,replay,resets)
            duplicate = rollout_intervention(env,observation.copy(),policy,actors[6000],critics[6000],'frozen_actor_6k',horizon=1)
            reconstruct_state(env,index,replay,resets)
            again = rollout_intervention(env,observation.copy(),policy,actors[6000],critics[6000],'frozen_actor_6k',horizon=1)
            assert duplicate == again
            membership = (['near'] if distance[index] < 1.2 else ['far'] if distance[index] >= 4 else []) + (['above'] if height[index] > .15 else ['below'] if height[index] < -.15 else [])
            result.append(dict(replay_index=index, transition_step=index+1, groups=membership, design_groups=labels,
                               goal_distance=float(distance[index]), height_error=float(height[index]),
                               velocity=(states[index,6:9]*2).tolist(), reset=reset,
                               reconstructed_observation_max_error=error, same_state_repeat_exact=True,
                               actor_action=policy.tolist(), actor_values=actor_values, physical_state_method='fresh reset plus exact real-action replay for every candidate', scans=scans))
            write(root/'paired_vertical_rollouts.json',result)
            print(f'[Paired Full] index={index} groups={labels} distance={distance[index]:.3f} height_error={height[index]:+.3f}',flush=True)
    finally:
        env.close()
    consistency = {}
    for group in groups:
        consistency[group] = {}
        for suffix in ('frozen_actor_6k','waypoint_height_feedback'):
            matching = [r for r in result if group in r['groups']]
            comparisons = []
            for record in matching:
                scans = [s for s in record['scans'] if s['suffix']==suffix]
                q1=np.array([s['q']['6000'][0] for s in scans])
                q2=np.array([s['q']['6000'][1] for s in scans])
                for horizon in ('30','90'):
                    returns=np.array([s['returns'][horizon]['discounted_scaled_return'] for s in scans])
                    for label, q in (('q1',q1),('q2',q2),('q_min',np.minimum(q1,q2))):
                        concordant=discordant=ties=0
                        for i in range(5):
                            for j in range(i+1,5):
                                product=(q[j]-q[i])*(returns[j]-returns[i])
                                if abs(returns[j]-returns[i])<1e-6 or abs(q[j]-q[i])<1e-7:
                                    ties+=1
                                elif product>0:
                                    concordant+=1
                                else:
                                    discordant+=1
                        comparisons.append(dict(index=record['replay_index'],horizon=int(horizon),critic=label,
                                                spearman=float(spearmanr(q,returns).statistic),concordant=concordant,discordant=discordant,ties=ties,
                                                q_top_z=float(ZGRID[np.argmax(q)]),return_top_z=float(ZGRID[np.argmax(returns)])))
            consistency[group][suffix]=comparisons
    write(root/'ranking_consistency.json',consistency)
    # Exact source mapping of the 297 success entries back into actual replay.
    episodes=rows(LOGS/'p0_audit/episodes.jsonl')
    success_ids=np.concatenate([np.arange(e['step']-e['length'],e['step']) for e in episodes if e['done_reason']=='success'])
    success=saved['success']['strata'][10]['arrays']
    assert len(success_ids)==saved['success']['strata'][10]['size']==297
    for name in success:
        np.testing.assert_array_equal(success[name],replay[name][success_ids])
    bias={}
    for label, arrays in (('success',success),('main',replay)):
        start_height=arrays['state'][:,2]*.5+1
        end_height=arrays['next_state'][:,2]*.5+1
        az=arrays['action'][:,2]
        bias[label]=dict(count=len(az),vertical_action=describe(az),positive_z_fraction=float(np.mean(az>0)),
                         z_saturation_fraction=float(np.mean(np.abs(az)>=.999)),height=describe(start_height),
                         height_delta=describe(end_height-start_height), actual_vertical_velocity=describe(arrays['state'][:,8]*2),
                         reward=describe(arrays['reward']),terminal_count=int((arrays['not_done']==0).sum()),
                         positive_reward_fraction=float(np.mean(arrays['reward']>0)))
    bias['successful_episode_composition']=[dict(step=e['step'],length=e['length'],return_raw=e['return'],
                                                reward_terms=e['reward_terms']) for e in episodes if e['done_reason']=='success']
    write(root/'success_replay_bias.json',bias)


def episode_eval(env, actor, observation):
    initial_position=env.pos[0].copy()
    minimum=float(np.linalg.norm(initial_position-env.TARGET_POS))
    tracking=[]
    samples=[]
    while True:
        with torch.no_grad():
            action=actor(torch.as_tensor(preprocess_state(observation)).reshape(1,-1))[0].numpy()
        observation,reward,terminated,truncated,info=env.step(action.reshape(1,3))
        minimum=min(minimum,info['goal_distance'])
        tracking.append(info['reference_tracking_error_mean'])
        samples.append(dict(action=action.tolist(), position=env.pos[0].tolist(), velocity=env.vel[0].tolist(),
                            reference_velocity=env.reference_velocity.tolist(), reference_position=env.reference_position.tolist(),
                            tracking_error=tracking[-1], reward=float(reward)))
        if terminated or truncated:
            break
    return dict(initial_position=initial_position.tolist(),initial_velocity=[0.,0.,0.],
                success=bool(info['success']),collision=bool(info['collision']),done_reason=info['done_reason'],
                length=len(samples),minimum_goal_distance=minimum,final_goal_distance=float(info['goal_distance']),
                guidance_entered=minimum<1.2,tracking_error_mean=float(np.mean(tracking)),
                tracking_error_max=float(max(tracking)),final_height=float(env.pos[0,2]),
                vertical_action_mean=float(np.mean([s['action'][2] for s in samples])),
                vertical_saturation_fraction=float(np.mean([abs(s['action'][2])>=.999 for s in samples])),samples=samples)


def evaluation(root):
    saved={step:load(step) for step in (2000,5000,6000)}
    assert fingerprint(saved[2000]['agent']['actor'])==fingerprint(saved[5000]['agent']['actor'])
    env=env_for(saved[6000],training_course=False,pool_split='validation')
    results={}
    try:
        for step in (2000,6000):
            results[str(step)]=[]
            actor=saved[step]['agent']['actor']
            for index in range(20):
                env.reset(seed=index)
                path=env.pool.path_for('validation',index,0)
                remaining=np.r_[np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1)[::-1])[::-1],0.]
                node=int(np.argmin(np.abs(remaining-.8)))
                assert .35<=remaining[node]<1.2
                observation=relocate_reset(env,path[node])
                record=episode_eval(env,actor,observation)
                record.update(map_id=env._pool_info()['map_id'],map_pool_index=index,map_split='validation',
                              start_remaining_path_length=float(remaining[node]),source_step=step,actor_sha256=fingerprint(actor))
                results[str(step)].append(record)
                write(root/'l0_validation.json',results)
                print(f'[L0 Eval] source={step} map={index} reason={record["done_reason"]} minimum={record["minimum_goal_distance"]:.3f}',flush=True)
    finally:
        env.close()
    results['5000']=copy.deepcopy(results['2000'])
    for record in results['5000']:
        record['source_step']=5000
        record['reused_identical_actor_2k']=True
    write(root/'l0_validation.json',results)
    # Existing original-start evaluations are already paired on the exact same
    # validation20; reuse immutable raw traces after verifying network identity.
    original={}
    for step,context in ((2000,'final_raw'),(5000,'periodic_raw'),(6000,'final_raw')):
        folder=next((LOGS/'validation'/context).glob(f'step{step}_source{step}_*'))
        data=json.loads((folder/'density0_metrics.json').read_text())
        assert data['actor_sha256']==fingerprint(saved[step]['agent']['actor'])
        original[str(step)]=[]
        for index, episode in enumerate(data['episodes']):
            with np.load(folder/'density0'/f'map{index:03d}.npz') as trace:
                distance=trace['lower_goal_distance']
                error=np.linalg.norm(trace['lower_position']-trace['lower_reference_position'],axis=1)
                actions=trace['upper_action']
                ep={**episode,'minimum_goal_distance':float(min(16.,distance.min())),
                    'guidance_entered':bool(distance.min()<1.2),'tracking_error_mean':float(error.mean()),
                    'tracking_error_max':float(error.max()),'vertical_action_mean':float(actions[:,2].mean()),
                    'vertical_saturation_fraction':float(np.mean(np.abs(actions[:,2])>=.999)),
                    'source_step':step,'source_metrics':str(folder/'density0_metrics.json'),'reused_verified_raw_trace':True}
            original[str(step)].append(ep)
    write(root/'original16m_validation.json',original)


def evaluation_summary(eps):
    output = {reason:sum(e['done_reason']==reason for e in eps) for reason in ('success','collision','height_bound','xy_bound','timeout')}
    output.update(episodes=len(eps),success_rate=output['success']/len(eps),collision_rate=output['collision']/len(eps),
                  guidance_fraction=float(np.mean([e['guidance_entered'] for e in eps])),
                  mean_minimum_distance=float(np.mean([e['minimum_goal_distance'] for e in eps])),
                  mean_final_distance=float(np.mean([e['final_goal_distance'] for e in eps])),
                  mean_tracking_error=float(np.mean([e['tracking_error_mean'] for e in eps])))
    return output


def summary(root):
    updates=json.loads((root/'update_audit.json').read_text())
    fine=json.loads((root/'fine_timing/update_audit.json').read_text())
    probes=json.loads((root/'timeline_probes.json').read_text())
    execution=json.loads((root/'executed_vs_actor.json').read_text())
    protected=json.loads((root/'protected_sources.json').read_text())
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha for path,sha in protected.items())
    five=load(5000)
    fixed=np.load(root/'fixed_states.npz')['states']
    dynamics=[]
    for step in range(5000,6001,100):
        actor=copy.deepcopy(five['agent']['actor'])
        actor.load_state_dict(torch.load(root/'checkpoints'/f'model_raw_{step}_actor',weights_only=True))
        targets=torch.load(root/'checkpoints'/f'targets_{step}.pt',weights_only=True)
        target_actor=copy.deepcopy(actor)
        target_actor.load_state_dict(targets['actor_target'])
        with torch.no_grad():
            x=torch.as_tensor(fixed)
            hidden=actor.policy[:-1](actor.encoder(x))
            logits=actor.policy[-1](hidden)
            actions=actor(x)
            target_actions=target_actor(x)
        dynamics.append(dict(step=step,actor_z_mean=float(actions[:,2].mean()),target_actor_z_mean=float(target_actions[:,2].mean()),
                             vertical_logit=describe(logits[:,2].numpy()),head_input_norm=describe(torch.linalg.vector_norm(hidden,dim=1).numpy()),
                             mean_tanh_z_derivative=float((1-actions[:,2]**2).mean()),
                             actor_target_action_rms=float(torch.sqrt(((actions-target_actions)**2).mean()))))
    write(root/'actor_head_and_target_lag.json',dynamics)
    first_gradient=next(r for r in fine if 'fixed_before_actor_update' in r and r['fixed_before_actor_update']['q1_z_gradient']['mean']>0)
    # Stable preference: at least ten consecutive actor updates with positive mean fixed-state dz.
    actor_updates=[r for r in fine if 'fixed_before_actor_update' in r]
    sustained=None
    for i in range(len(actor_updates)-9):
        if all(r['fixed_before_actor_update']['q1_z_gradient']['mean']>0 for r in actor_updates[i:i+10]):
            sustained=actor_updates[i]['step'];break
    timing=dict(first_positive_fixed_q1_dz=first_gradient['step'],sustained_positive_fixed_q1_dz=sustained,
                first_policy_mean_z_05=next(r['step'] for r in updates if r['fixed_z_mean_after_update']>=.5),
                first_policy_mean_z_09=next(r['step'] for r in updates if r['fixed_z_mean_after_update']>=.9),
                first_policy_mean_z_099=next(r['step'] for r in updates if r['fixed_z_mean_after_update']>=.99),
                first_all_fixed_states_z_saturated=next(r['step'] for r in updates if r['fixed_z_saturation_after_update']==1),
                negative_minimum_step=min(updates,key=lambda r:r['fixed_z_mean_after_update'])['step'],
                negative_minimum_z=min(r['fixed_z_mean_after_update'] for r in updates))
    write(root/'collapse_timing.json',timing)
    original=json.loads((root/'original16m_validation.json').read_text())
    l0=json.loads((root/'l0_validation.json').read_text())
    evals={label:{step:evaluation_summary(eps) for step,eps in data.items()} for label,data in (('original16m',original),('L0',l0))}
    write(root/'evaluation_summary.json',evals)
    tracking=[]
    folder=next((LOGS/'validation/final_raw').glob('step6000_source6000_*'))
    for index,ep in enumerate(original['6000']):
        with np.load(folder/'density0'/f'map{index:03d}.npz') as trace:
            actual=trace['lower_velocity'][:,2]
            ref=trace['lower_reference_velocity'][:,2]
            command=trace['lower_command_velocity'][:,2]
            z=trace['lower_position'][:,2]
            tracking.append(dict(index=index,done_reason=ep['done_reason'],initial_velocity=trace['lower_velocity'][0].tolist(),
                                 command_z_mean=float(command.mean()),last_actual_vz=float(actual[-1]),
                                 last_reference_vz=float(ref[-1]),vertical_velocity_tracking_rmse=float(np.sqrt(np.mean((actual-ref)**2))),
                                 final_z=float(z[-1]),max_abs_roll_pitch=float(np.abs(trace['lower_rpy'][:,:2]).max()),
                                 duration_seconds=float(trace['lower_time_seconds'][-1]),
                                 height_crossing_sign='upper' if z[-1]>2.75 else 'lower' if z[-1]<.08 else None))
    write(root/'height_failure_tracking.json',tracking)
    value_root = root / 'values_rebuilt' if (root / 'values_rebuilt' / 'ranking_consistency.json').exists() else root
    rank_path = value_root/'ranking_consistency_world_groups.json'
    if not rank_path.exists():
        rank_path = value_root/'ranking_consistency.json'
    rank=json.loads(rank_path.read_text())
    rank_summary={}
    for group,suffixes in rank.items():
        rank_summary[group]={}
        for suffix,records in suffixes.items():
            rank_summary[group][suffix]={}
            for critic in ('q1','q2','q_min'):
                for horizon in (30,90):
                    selected=[r for r in records if r['critic']==critic and r['horizon']==horizon]
                    c=sum(r['concordant'] for r in selected);d=sum(r['discordant'] for r in selected)
                    rank_summary[group][suffix][f'{critic}_H{horizon}']=dict(states=len(selected),concordant=c,discordant=d,
                        ties=sum(r['ties'] for r in selected),concordance_fraction=c/(c+d) if c+d else None,
                        mean_spearman=float(np.mean([r['spearman'] for r in selected])),
                        top_action_agreement=sum(r['q_top_z']==r['return_top_z'] for r in selected))
    write(root/'ranking_summary.json',rank_summary)
    samples=[]
    replay=load(6000)['replay']['arrays']
    for row in execution:
        if row['done_reason']=='height_bound':
            index=row['step']-1
            z=float(replay['next_state'][index,2]*.5+1)
            samples.append(dict(step=row['step'],terminal_z=z,side='upper' if z>2.75 else 'lower',
                                deterministic_z=row['deterministic_action'][2],executed_z=row['executed_action'][2]))
    write(root/'training_height_failures.json',samples)
    paired=json.loads((value_root/'paired_vertical_rollouts.json').read_text())
    report=dict(replication=json.loads((root/'replication.json').read_text()),
                fine_replication=json.loads((root/'fine_timing/replication.json').read_text()),
                timing=timing,evaluations=evals,ranking=rank_summary,training_height_failures=samples,
                success_bias=json.loads((value_root/'success_replay_bias.json').read_text()), value_results_path=str(value_root),
                actor_dynamics=dynamics,td_audit=dict(updates_checked=len(updates),main_count=205,success_count=51,
                                                    maximum_reconstructed_loss_error=max(r['reconstruction_absolute_error'] for r in updates)),
                physical_states=len(paired),all_state_reconstruction_exact=all(r['reconstructed_observation_max_error']==0 for r in paired),
                all_paired_state_repeats_exact=all(r['same_state_repeat_exact'] for r in paired),
                source_files_unchanged=len(protected),test_policy_evaluations=0,
                historical_harness_error='Earlier Full physical probe helpers inherited K1 integration_dt=8/120. This diagnostic harness uses the original Full 1/120 and matches 3000 real Replay rewards. Production training reward is unchanged.',
                limitations=['Finite-horizon deterministic-policy returns are not exact infinite-horizon smoothed-target TD3 Q truth.',
                             '12 representative train states selected with 3 per design stratum; final world-distance/height groups overlap. Limited statistical support, tied returns excluded from concordance.',
                             'Waypoint/height feedback suffix differs from the learned policy; its ranking only diagnoses recoverability.',
                             '5k and 2k actors are hash-identical; their L0 result is deliberately reused after identity check. Original16m uses verified existing raw trajectories.',
                             'First fixed-state saturation time is conditional on this preregistered state set, not all reachable states.'])
    write(root/'report.json',report)
    print(json.dumps(dict(timing=timing,evaluations=evals,source_files_unchanged=len(protected)),indent=2))


def plot_audit(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    updates=json.loads((root/'update_audit.json').read_text())
    fine=json.loads((root/'fine_timing/update_audit.json').read_text())
    execution=json.loads((root/'executed_vs_actor.json').read_text())
    fig,axes=plt.subplots(4,1,figsize=(10,11),sharex=True,constrained_layout=True)
    steps=[r['step'] for r in updates]
    axes[0].plot(steps,[r['fixed_z_mean_after_update'] for r in updates],label='Fixed-state mean actor az')
    axes[0].plot(steps,[r['fixed_z_saturation_after_update'] for r in updates],label='Fixed-state vertical saturation',alpha=.75)
    axes[0].axvline(5125,color='red',linestyle='--',alpha=.5)
    axes[0].set_ylabel('Normalized action / fraction');axes[0].legend()
    selected=[r for r in fine if 'fixed_before_actor_update' in r]
    axes[1].plot([r['step'] for r in selected],[r['fixed_before_actor_update']['q1_z_gradient']['mean'] for r in selected],label='Fixed-state dQ1/daz before actor update')
    axes[1].plot([r['step'] for r in selected],[r['actor_batch_q1_action_gradient_mean'][2] for r in selected],label='Actual minibatch dQ1/daz',alpha=.7)
    axes[1].axhline(0,color='gray',linewidth=.8);axes[1].set_ylabel('Q gradient');axes[1].legend()
    axes[2].plot([r['step'] for r in execution],[r['executed_action'][2] for r in execution],label='Executed az with exploration',alpha=.45)
    axes[2].plot([r['step'] for r in execution],[r['deterministic_action'][2] for r in execution],label='Deterministic actor az',linewidth=1)
    axes[2].set_ylabel('Normalized az');axes[2].legend()
    axes[3].plot([r['step'] for r in execution],[r['height'] for r in execution],label='Actual episode height (reset discontinuities)')
    axes[3].axhline(2.75,color='red',linestyle=':');axes[3].axhline(.08,color='red',linestyle=':')
    axes[3].set_ylabel('Height (m)');axes[3].set_xlabel('Cumulative upper transitions');axes[3].legend()
    fig.suptitle('Exact isolated 5k-6k replication: critic gradient leads actor saturation')
    fig.savefig(root/'collapse_timeline.png',dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('part', choices=('timeline', 'fine_timing', 'values', 'evaluation', 'summary'))
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.part in ('timeline','fine_timing'):
        args.output.mkdir(parents=True, exist_ok=False)
        protected = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in SOURCE.iterdir() if path.is_file()}
        write(args.output / 'protected_sources.json', protected)
        timeline(args.output, final_step=5200 if args.part=='fine_timing' else 6000)
    else:
        globals()[args.part](args.output)
        if args.part == "summary":
            plot_audit(args.output)


if __name__ == '__main__':
    main()
