"""Isolated 16m contract candidates; never registered as production modes.

Input variants require fresh learning and consistently encoded Replay. Existing
Actor weights must not be evaluated on a different encoding as a learning test.
"""
from dataclasses import dataclass
import copy
from types import MethodType

import numpy as np
import torch

from envs.preprocess import preprocess_state
from scripts.audit_ego20x10_value import env_for
from algos.td3.td3_upper_recurrent import _project_planar_action_tensor


@dataclass(frozen=True)
class Contract:
    name: str
    goal_range: bool = False
    position_scale: bool = False
    guidance_project: bool = False
    gamma: float = .99


CONTRACTS = {
    c.name: c for c in (
        Contract('baseline'), Contract('goal_range', goal_range=True),
        Contract('position_scale', position_scale=True),
        Contract('guidance_project', guidance_project=True),
        Contract('gamma997', gamma=.997),
    )
}
WORLD_SPEED = np.array([.8, .8, .3], dtype=np.float32)
GOAL_SCALE = np.array([20., 10., 2.], dtype=np.float32)


def encode_observation(observation, contract):
    encoded = preprocess_state(observation)
    if encoded.shape != (917,):
        raise ValueError('This isolated experiment requires the existing 917D layout')
    if contract.position_scale:
        raw = np.asarray(observation, dtype=np.float32).reshape(-1)
        encoded[:3] = (raw[:3] - [0., 0., 1.]) / [10., 5., 1.]
    return encoded


def reencode_replay_state(states, contract, target=(8., 0., 1.)):
    """Change a known baseline encoding, recovering unclipped world position.

    Rewards must be rebuilt separately for guidance_project; this function does
    not claim to convert that variant's full transition contract.
    """
    result = np.array(states, dtype=np.float32, copy=True)
    position = result[..., :3] * .5 + np.array([0., 0., 1.])
    if contract.goal_range:
        result[..., 12:15] = np.clip((np.asarray(target)-position)/GOAL_SCALE, -1., 1.)
    if contract.position_scale:
        result[..., :3] = (position-[0., 0., 1.])/[10., 5., 1.]
    return result


def desired_velocity(model, position, target, tolerance=.2):
    delta = np.asarray(target, dtype=np.float32)-np.asarray(position, dtype=np.float32)
    distance = float(np.linalg.norm(delta))
    ratio = float(np.clip((distance-tolerance)/(model.near_goal_guidance_radius-tolerance), 0., 1.))
    speed = model.near_goal_capture_speed + ratio*(model.near_goal_cruise_speed-model.near_goal_capture_speed)
    return speed*delta/max(distance, 1e-6)


def project_velocity(velocity):
    projected = np.array(velocity, dtype=np.float32, copy=True)
    projected[:2] *= min(1., .8/max(float(np.linalg.norm(projected[:2])), 1e-9))
    projected[2] = np.clip(projected[2], -.3, .3)
    return projected


class FeasibleGuidanceReward:
    """Delegate all original terms, replace just the tracking target vector."""
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def compute(self, **kwargs):
        reward, terms, distance = self.original.compute(**kwargs)
        if kwargs.get('terminal_reason', 'running') != 'running' or distance >= self.near_goal_guidance_radius or distance <= 1e-6:
            return reward, terms, distance
        state = np.asarray(kwargs['state'], dtype=np.float32)
        old = desired_velocity(self.original, state[:3], kwargs['target_pos'], kwargs['goal_tolerance'])
        new = project_velocity(old)
        # Preserve even floating point roundoff when the target is already feasible.
        if np.array_equal(old, new):
            return reward, terms, distance
        activation = np.clip((self.near_goal_guidance_radius-distance)/self.near_goal_guidance_ramp_width, 0., 1.)
        penalty = float(self.near_goal_velocity_tracking_weight*activation*np.sum((state[10:13]-new)**2)*self.integration_dt)
        difference = penalty-terms['near_goal_velocity_tracking_penalty']
        terms = dict(terms)
        terms['near_goal_velocity_tracking_penalty'] = penalty
        terms['continuous_penalty_total'] += difference
        terms['running_shaping_reward'] -= difference
        terms['near_goal_desired_speed'] = float(np.linalg.norm(new))
        return float(reward-difference), terms, distance


def attach_contract(env, contract):
    """Modify only this diagnostic instance before reset; no global patching."""
    if contract.goal_range:
        def goal(base, position):
            delta = base.TARGET_POS-np.asarray(position, dtype=np.float32)
            return np.clip(delta/GOAL_SCALE, -1., 1.).astype(np.float32)
        env.base_env._computeGoalObservation = MethodType(goal, env.base_env)
    if contract.guidance_project:
        env.base_env._reward_model = FeasibleGuidanceReward(env.base_env._reward_model)
    env.diagnostic_contract = contract
    return env


def contract_env(payload, contract, *, split='train'):
    if split not in ('train', 'validation'):
        raise ValueError('The sealed test split is outside this experiment')
    return attach_contract(env_for(payload, training_course=False, pool_split=split), contract)


def action_change(actor, states, previous_actions):
    with torch.no_grad():
        actions = actor(states).detach()
        delta = torch.linalg.vector_norm((actions-previous_actions)*states.new_tensor(WORLD_SPEED), dim=1)
    return {'mean_mps': float(delta.mean()), 'p95_mps': float(torch.quantile(delta, .95)),
            'max_mps': float(delta.max())}


class ActorStepGuard:
    """A transaction around the existing TD3 train call, in world velocity.

    Rejected steps restore Actor, target Actor and Adam; critic updates remain.
    Backtracking replays the same Adam gradient at a smaller LR, preserving
    accepted Adam moments. It bounds one step, not cumulative drift or safety.
    """
    def __init__(self, agent, states, mean_limit=.02, p95_limit=.05, backtracks=10):
        self.agent = agent
        self.states = torch.as_tensor(states, dtype=torch.float32, device=agent.device)
        if not len(self.states) or not torch.isfinite(self.states).all():
            raise ValueError('Guard requires a finite, nonempty training-state set')
        if mean_limit < 0 or p95_limit < 0 or backtracks < 0:
            raise ValueError('Guard bounds and backtrack count must be non-negative')
        self.mean_limit, self.p95_limit, self.backtracks = mean_limit, p95_limit, backtracks

    def acceptable(self, metrics):
        return (np.isfinite(list(metrics.values())).all()
                and metrics['mean_mps'] <= self.mean_limit and metrics['p95_mps'] <= self.p95_limit)

    def train(self, *args, **kwargs):
        agent = self.agent
        scheduled = (agent.total_it+1) % agent.policy_freq == 0 and not agent.actor_frozen
        if not scheduled:
            return agent.train(*args, **kwargs), None
        before_actor = copy.deepcopy(agent.actor.state_dict())
        before_target = copy.deepcopy(agent.actor_target.state_dict())
        before_adam = copy.deepcopy(agent.actor_optimizer.state_dict())
        with torch.no_grad():
            before_actions = agent.actor(self.states).detach().clone()
        info = agent.train(*args, **kwargs)
        raw = action_change(agent.actor, self.states, before_actions)
        metrics, scale = raw, 1.
        accepted = self.acceptable(metrics)
        for attempt in range(1, self.backtracks+1):
            if accepted:
                break
            scale = .5**attempt
            agent.actor.load_state_dict(before_actor)
            agent.actor_optimizer.load_state_dict(before_adam)
            lrs = [g['lr'] for g in agent.actor_optimizer.param_groups]
            for group, lr in zip(agent.actor_optimizer.param_groups, lrs):
                group['lr'] = lr*scale
            try:
                agent.actor_optimizer.step()
            finally:
                for group, lr in zip(agent.actor_optimizer.param_groups, lrs):
                    group['lr'] = lr
            metrics = action_change(agent.actor, self.states, before_actions)
            accepted = self.acceptable(metrics)
        agent.actor_target.load_state_dict(before_target)
        if accepted:
            with torch.no_grad():
                for parameter, target in zip(agent.actor.parameters(), agent.actor_target.parameters()):
                    target.copy_(agent.tau*parameter+(1-agent.tau)*target)
        else:
            agent.actor.load_state_dict(before_actor)
            agent.actor_optimizer.load_state_dict(before_adam)
            scale = 0.
            metrics = action_change(agent.actor, self.states, before_actions)
        info['actor_updated'] = bool(accepted)
        info['actor_proposal_made'] = True
        return info, {'accepted': bool(accepted), 'lr_scale': scale, 'raw_change': raw,
                      'accepted_change': metrics, 'mean_limit_mps': self.mean_limit, 'p95_limit_mps': self.p95_limit}


def critic_calibration_step(agent, batch, pairs=None, weight=0.):
    """Frozen-Actor TD3 critic update plus local finite-return differences.

    Exactly one Adam step: L_TD + weight * sum_j MSE(delta Q_j, delta G).
    Paired finite-horizon returns are a diagnostic surrogate, not infinite-
    horizon TD3 truth. Only train-map pairs may be supplied to this update.
    """
    if not agent.actor_frozen:
        raise ValueError('Local calibration pilot requires a frozen Actor')
    if weight < 0 or (weight > 0 and pairs is None):
        raise ValueError('Positive calibration weight requires paired labels')
    agent.total_it += 1
    state, action, next_state, reward, mask = [x.to(agent.device) for x in batch]
    with torch.no_grad():
        noise = (torch.randn_like(action)*agent.policy_noise).clamp(-agent.noise_clip, agent.noise_clip)
        next_action = (agent.actor_target(next_state)+noise).clamp(-agent.max_action, agent.max_action)
        if agent.project_planar_action:
            next_action = _project_planar_action_tensor(next_action, agent.max_action)
        q1, q2 = agent.critic_target(next_state, next_action)
        target = reward+mask*agent.discount*torch.minimum(q1, q2)
    current = agent.critic(state, action)
    td_loss = sum(torch.nn.functional.mse_loss(q, target) for q in current)
    paired_loss = state.new_zeros(())
    if weight > 0:
        s, plus, minus, difference = [x.to(agent.device) for x in pairs]
        qp = agent.critic(s, plus)
        qm = agent.critic(s, minus)
        paired_loss = sum(torch.nn.functional.mse_loss(a-b, difference) for a, b in zip(qp, qm))
    loss = td_loss+weight*paired_loss
    agent.critic_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), agent.grad_clip_norm)
    agent.critic_optimizer.step()
    if agent.total_it % agent.policy_freq == 0:
        with torch.no_grad():
            for parameter, target_parameter in zip(agent.critic.parameters(), agent.critic_target.parameters()):
                target_parameter.copy_(agent.tau*parameter+(1-agent.tau)*target_parameter)
    return {'td_loss': float(td_loss.detach()), 'paired_loss': float(paired_loss.detach()),
            'combined_loss': float(loss.detach()), 'critic_updates': agent.total_it, 'actor_updated': False}
