
import os
import time
import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


def _grad_norm(parameters) -> float:
	total_sq = 0.0
	for p in parameters:
		if p.grad is None:
			continue
		g = p.grad.detach()
		total_sq += float(torch.sum(g * g).item())
	return float(total_sq ** 0.5)


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class Encoder(nn.Module):
	def __init__(self, state_dim, latent_dim=16):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(state_dim, 128),
			nn.ReLU(),
			nn.Linear(128, latent_dim),
		)

	def forward(self, s):
		return self.net(s)


class ReconHead(nn.Module):
	def __init__(self, latent_dim=16, state_dim=12):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(latent_dim, 64),
			nn.ReLU(),
			nn.Linear(64, state_dim),
		)

	def forward(self, z):
		return self.net(z)


class DynHead(nn.Module):
	def __init__(self, latent_dim=16, action_dim=4, state_dim=12):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(latent_dim + action_dim, 64),
			nn.ReLU(),
			nn.Linear(64, state_dim),
		)

	def forward(self, z, a):
		return self.net(torch.cat([z, a], dim=1))


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		actor_updates_encoder=True,
		latent_input_scale=0.1,
		grad_clip_norm=1.0,
	):
		latent_dim = 16
		self.latent_dim = latent_dim
		new_state_dim = state_dim + latent_dim

		self.encoder = Encoder(state_dim, latent_dim=latent_dim).to(device)
		self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=3e-4)
		self.recon_head = ReconHead(latent_dim=latent_dim, state_dim=12).to(device)
		self.recon_optimizer = torch.optim.Adam(self.recon_head.parameters(), lr=1e-3)
		self.dyn_head = DynHead(latent_dim=latent_dim, action_dim=action_dim, state_dim=12).to(device)
		self.dyn_optimizer = torch.optim.Adam(self.dyn_head.parameters(), lr=1e-3)

		self.actor = Actor(new_state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(new_state_dim, action_dim).to(device)
		self.critic_target = Critic(new_state_dim, action_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.actor_updates_encoder = bool(actor_updates_encoder)
		self.latent_input_scale = float(latent_input_scale)
		self.grad_clip_norm = float(grad_clip_norm)

		self.total_it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		with torch.no_grad():
			latent = self.encoder(state)
			state_input = torch.cat([state, latent * self.latent_input_scale], dim=1)
		return self.actor(state_input).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		state = torch.FloatTensor(state).to(device)
		next_state = torch.FloatTensor(next_state).to(device)
		latent = self.encoder(state)
		latent_next = self.encoder(next_state)
		state_input = torch.cat([state, latent * self.latent_input_scale], dim=1)
		next_state_input = torch.cat([next_state, latent_next * self.latent_input_scale], dim=1)
		state_12d = state[:, :12]
		next_state_12d = next_state[:, :12]
		s_hat = self.recon_head(latent)
		recon_loss = F.mse_loss(s_hat, state_12d)
		s_next_hat = self.dyn_head(latent, action)
		dyn_loss = F.mse_loss(s_next_hat, next_state_12d)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state_input) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state_input, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state_input, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		loss = critic_loss + 0.1 * recon_loss + 0.1 * dyn_loss

		# Optimize the critic
		self.encoder_optimizer.zero_grad()
		self.recon_optimizer.zero_grad()
		self.dyn_optimizer.zero_grad()
		self.critic_optimizer.zero_grad()
		loss.backward()
		encoder_grad_norm_critic_recon = _grad_norm(self.encoder.parameters())
		torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.grad_clip_norm)
		torch.nn.utils.clip_grad_norm_(self.recon_head.parameters(), self.grad_clip_norm)
		torch.nn.utils.clip_grad_norm_(self.dyn_head.parameters(), self.grad_clip_norm)
		torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
		self.encoder_optimizer.step()
		self.recon_optimizer.step()
		self.dyn_optimizer.step()
		self.critic_optimizer.step()


		actor_loss_val = None
		encoder_grad_norm_actor = 0.0
		actor_updated = False

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			if self.actor_updates_encoder:
				latent_actor = self.encoder(state)
			else:
				latent_actor = self.encoder(state).detach()
			state_input_actor = torch.cat([state, latent_actor * self.latent_input_scale], dim=1)
			actor_loss = -self.critic.Q1(state_input_actor, self.actor(state_input_actor)).mean()
			
			# Optimize the actor 
			self.encoder_optimizer.zero_grad()
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			encoder_grad_norm_actor = _grad_norm(self.encoder.parameters()) if self.actor_updates_encoder else 0.0
			if self.actor_updates_encoder:
				torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.grad_clip_norm)
				self.encoder_optimizer.step()
			torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
			self.actor_optimizer.step()
			actor_loss_val = float(actor_loss.item())
			actor_updated = True

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		return {
			"critic_loss": float(critic_loss.item()),
			"recon_loss": float(recon_loss.item()),
			"dyn_loss": float(dyn_loss.item()),
			"encoder_grad_norm_critic_recon": float(encoder_grad_norm_critic_recon),
			"encoder_grad_norm_actor": float(encoder_grad_norm_actor),
			"latent_std_mean": float(latent.detach().std(dim=0).mean().item()),
			"actor_loss": actor_loss_val,
			"actor_updated": actor_updated,
		}

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		torch.save(self.encoder.state_dict(), filename + "_encoder")
		torch.save(self.encoder_optimizer.state_dict(), filename + "_encoder_optimizer")
		torch.save(self.recon_head.state_dict(), filename + "_recon_head")
		torch.save(self.recon_optimizer.state_dict(), filename + "_recon_optimizer")
		torch.save(self.dyn_head.state_dict(), filename + "_dyn_head")
		torch.save(self.dyn_optimizer.state_dict(), filename + "_dyn_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)
		self.encoder.load_state_dict(torch.load(filename + "_encoder"))
		self.encoder_optimizer.load_state_dict(torch.load(filename + "_encoder_optimizer"))
		self.recon_head.load_state_dict(torch.load(filename + "_recon_head"))
		self.recon_optimizer.load_state_dict(torch.load(filename + "_recon_optimizer"))
		self.dyn_head.load_state_dict(torch.load(filename + "_dyn_head"))
		self.dyn_optimizer.load_state_dict(torch.load(filename + "_dyn_optimizer"))

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = int(max_size)
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
		self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
		self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
		self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
		self.not_done = np.zeros((self.max_size, 1), dtype=np.float32)

	def push(self, state, action, reward, next_state, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = float(reward)
		self.not_done[self.ptr] = 1.0 - float(done)

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		return (
			torch.as_tensor(self.state[ind], device=device),
			torch.as_tensor(self.action[ind], device=device),
			torch.as_tensor(self.next_state[ind], device=device),
			torch.as_tensor(self.reward[ind], device=device),
			torch.as_tensor(self.not_done[ind], device=device),
		)


def _preprocess_state(state_1d: np.ndarray) -> np.ndarray:
	"""
	HoverAviary(ObservationType.KIN) first 12 dimensions are kinematics (pos/rpy/vel/ang_vel),
	the rest are action buffer (already normalized [-1, 1] actions), so only normalize first 12 dimensions for stability.
	"""
	state = np.asarray(state_1d, dtype=np.float32).reshape(-1)
	if state.shape[0] < 12:
		raise ValueError(f"Expected state dim >= 12, got {state.shape[0]}")

	obs_mean = np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32)
	obs_std = np.array([0.5, 0.5, 0.5, np.pi, np.pi, np.pi, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0], dtype=np.float32)
	obs_std[obs_std == 0] = 1.0

	kin = state[:12]
	kin_norm = (kin - obs_mean) / obs_std
	if state.shape[0] == 12:
		return kin_norm.astype(np.float32)
	return np.concatenate([kin_norm, state[12:]], axis=0).astype(np.float32)


def evaluate_policy(env: gym.Env, agent: TD3, preprocess_state_fn, episodes: int = 3) -> float:
	returns = []
	for ep in range(episodes):
		obs, _ = env.reset(seed=ep)
		state = preprocess_state_fn(obs.reshape(-1))
		done = False
		ep_ret = 0.0
		while not done:
			action = agent.select_action(state)
			obs2, reward, terminated, truncated, _ = env.step(action.reshape(1, -1))
			state = preprocess_state_fn(obs2.reshape(-1))
			done = bool(terminated or truncated)
			ep_ret += float(reward)
		returns.append(ep_ret)
	return float(np.mean(returns))


def _compute_actor_sat_pct(agent: TD3, states: torch.Tensor, eps: float = 1e-3) -> float:
	with torch.no_grad():
		latent = agent.encoder(states)
		state_input = torch.cat([states, latent * agent.latent_input_scale], dim=1)
		a_pi = agent.actor(state_input)
		# tanh+max_action is theoretically always within bounds; here we define saturation as "close to boundary"
		sat = (a_pi.abs() >= (agent.max_action * (1.0 - eps))).float().mean().item()
		return float(sat * 100.0)


def _compute_q_mean_and_actor_stats(agent: TD3, replay_buffer: ReplayBuffer, batch_size: int):
	if replay_buffer.size < batch_size:
		return None, None
	states, _, _, _, _ = replay_buffer.sample(batch_size)
	with torch.no_grad():
		latent = agent.encoder(states)
		state_input = torch.cat([states, latent * agent.latent_input_scale], dim=1)
		a_pi = agent.actor(state_input)
		q1, q2 = agent.critic(state_input, a_pi)
		q_min_mean = torch.min(q1, q2).mean().item()
		actor_sat_pct = _compute_actor_sat_pct(agent, states)
	return float(q_min_mean), float(actor_sat_pct)


def _physics_monitor_from_env(env: HoverAviary, last_action: np.ndarray):
	"""
	Physics monitor based on HoverAviary / BaseAviary internal state.
	Note: last_action is the [-1, 1] normalized RPM action output by the actor/exploration (one dimension per motor).
	"""
	target = np.array([0.0, 0.0, 1.0], dtype=np.float32)
	pos = env.pos[0].astype(np.float32)
	rpy = env.rpy[0].astype(np.float32)  # radians
	vel = env.vel[0].astype(np.float32)

	# ActionType.RPM mapping in BaseRLAviary:
	# rpm = HOVER_RPM * (1 + 0.05 * target)
	# last_action should be of shape (4,)
	last_action = np.asarray(last_action, dtype=np.float32).reshape(-1)
	motor_rpms = (env.HOVER_RPM * (1.0 + 0.05 * last_action)).astype(np.float32)
	motor_rpms = np.clip(motor_rpms, 0.0, env.MAX_RPM).astype(np.float32)
	sat_pct = (motor_rpms / env.MAX_RPM * 100.0).astype(np.float32)

	front_mean = float(motor_rpms[[0, 1]].mean())
	back_mean = float(motor_rpms[[2, 3]].mean())
	motor_front_back_diff = abs(front_mean - back_mean)

	action_sat = float((np.abs(last_action) >= (1.0 - 1e-3)).mean() * 100.0)

	# Thrust: forces = rpm^2 * KF
	avg_thrust = float((motor_rpms.astype(np.float64) ** 2 * float(env.KF)).mean())

	dist = float(np.linalg.norm(pos - target))
	r_deg, p_deg, y_deg = (rpy * env.RAD2DEG).astype(np.float32).tolist()
	z_vel = float(vel[2])

	return {
		"pos": pos,
		"rpy_deg": (r_deg, p_deg, y_deg),
		"z_vel": z_vel,
		"dist": dist,
		"motor_rpms": motor_rpms,
		"sat_pct": sat_pct,
		"motor_front_back_diff": motor_front_back_diff,
		"action_sat_pct": action_sat,
		"avg_thrust": avg_thrust,
	}


def evaluate_policy_with_gui_monitor(
	env_kwargs: dict,
	agent: TD3,
	preprocess_state_fn,
	replay_buffer: ReplayBuffer,
	episodes: int,
	gui: bool,
	eval_stepsleep: bool,
	print_debug: bool,
	train_steps_this_tick: int,
	q_batch_size: int,
):
	"""
	Each evaluation creates a separate env with gui=True, and closes the GUI automatically after completion.
	"""
	returns = []
	last_monitor = None

	q_mean, actor_sat_pct = _compute_q_mean_and_actor_stats(agent, replay_buffer, q_batch_size)

	for ep in range(episodes):
		# Create new env for each evaluation: ensure GUI is only open during evaluation
		env = HoverAviary(**env_kwargs, gui=gui)
		obs, _ = env.reset(seed=ep)
		state = preprocess_state_fn(obs.reshape(-1))
		done = False
		ep_ret = 0.0
		last_action = None
		start_time = time.time()
		step_idx = 0
		while not done:
			action = agent.select_action(state)
			last_action = action.copy()
			obs2, reward, terminated, truncated, _ = env.step(action.reshape(1, -1))
			state = preprocess_state_fn(obs2.reshape(-1))
			done = bool(terminated or truncated)
			ep_ret += float(reward)
			step_idx += 1
			if eval_stepsleep:
				sync(step_idx, start_time, env.CTRL_TIMESTEP)

		# Physics monitor for the last frame
		last_monitor = _physics_monitor_from_env(env, last_action)
		returns.append(ep_ret)
		env.close()

	# Ensure returned q_mean / actor_sat_pct are consistent with replay_buffer sampling
	if q_mean is None:
		q_mean = float("nan")
	if actor_sat_pct is None:
		actor_sat_pct = float("nan")

	return float(np.mean(returns)), q_mean, actor_sat_pct, last_monitor


def train(args):
	os.makedirs(args.output_dir, exist_ok=True)
	run_id = f"TD3_Hover_seed{args.seed}_{int(time.time())}"
	log_dir = os.path.join(args.output_dir, run_id)
	os.makedirs(log_dir, exist_ok=True)

	writer = SummaryWriter(log_dir=log_dir)

	rng = np.random.default_rng(args.seed)

	# Initial position in air (recommended random height: 1.0~1.5m), keep attitude level
	INIT_XYZS = np.array([[0.0, 0.0, rng.uniform(1.0, 1.5)]], dtype=np.float32)
	INIT_RPYS = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

	env = HoverAviary(
		obs=ObservationType.KIN,
		act=ActionType.RPM,
		initial_xyzs=INIT_XYZS,
		initial_rpys=INIT_RPYS,
		gui=args.gui,
		pyb_freq=args.pyb_freq,
		ctrl_freq=args.ctrl_freq,
	)

	# HoverAviary(KIN) obs is usually (NUM_DRONES, 12 + ACTION_BUFFER_SIZE*4).
	# For single drone, it's (1, state_dim_flat), so use flattened dimension here.
	state_dim = int(np.prod(env.observation_space.shape))
	action_dim = int(env.action_space.shape[-1])
	max_action = float(env.action_space.high.flatten()[0])
	print(f"[INFO] state_dim={state_dim}, action_dim={action_dim}, max_action={max_action}")

	agent = TD3(
		state_dim=state_dim,
		action_dim=action_dim,
		max_action=max_action,
		discount=args.gamma,
		tau=args.tau,
		policy_noise=args.policy_noise,
		noise_clip=args.noise_clip,
		policy_freq=args.policy_freq,
		actor_updates_encoder=args.actor_updates_encoder,
		latent_input_scale=args.latent_input_scale,
		grad_clip_norm=args.grad_clip_norm,
	)

	buffer = ReplayBuffer(state_dim, action_dim, max_size=args.buffer_size)

	# Make training reproducible (not fully deterministic)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	state, _ = env.reset(seed=args.seed)
	state = _preprocess_state(state.reshape(-1))

	best_eval_return = -1e18
	episode_return = 0.0
	episode_step = 0
	train_steps_this_tick = 0

	for t in range(1, args.total_steps + 1):
		# Exploration noise decay
		expl_noise = max(args.expl_noise_end,
			args.expl_noise_start - (t / args.noise_decay_steps) * (args.expl_noise_start - args.expl_noise_end)
		)

		if t < args.start_timesteps:
			action = rng.uniform(-max_action, max_action, size=(action_dim,)).astype(np.float32)
		else:
			action = agent.select_action(state)
			action = action + expl_noise * rng.normal(size=action_dim)
			action = np.clip(action, -max_action, max_action).astype(np.float32)

		next_obs, reward, terminated, truncated, _ = env.step(action.reshape(1, -1))
		done = bool(terminated or truncated)
		scaled_reward = float(reward) * args.reward_scale

		next_state = _preprocess_state(next_obs.reshape(-1))
		buffer.push(state, action, scaled_reward, next_state, done)

		state = next_state
		episode_return += float(reward)
		episode_step += 1

		if t >= args.update_after and buffer.size >= args.batch_size and t % args.train_every == 0:
			train_info = agent.train(buffer, batch_size=args.batch_size)
			train_steps_this_tick += 1
			# Record training loss (for tensorboard viewing)
			writer.add_scalar("Loss/critic_loss", train_info["critic_loss"], t)
			writer.add_scalar("Loss/recon_loss", train_info["recon_loss"], t)
			writer.add_scalar("Loss/dyn_loss", train_info["dyn_loss"], t)
			writer.add_scalar("Grad/encoder_from_critic_recon", train_info["encoder_grad_norm_critic_recon"], t)
			writer.add_scalar("Grad/encoder_from_actor", train_info["encoder_grad_norm_actor"], t)
			writer.add_scalar("Latent/std_mean", train_info["latent_std_mean"], t)
			if train_info["actor_loss"] is not None:
				writer.add_scalar("Loss/actor_loss", train_info["actor_loss"], t)

		# Episode end
		if done:
			writer.add_scalar("Reward/Train_EpisodeReturn", episode_return, t)
			writer.add_scalar("Stats/EpisodeLength", episode_step, t)

			episode_return = 0.0
			episode_step = 0
			next_obs, _ = env.reset()
			state = _preprocess_state(next_obs.reshape(-1))

		# Periodic evaluation
		if t % args.eval_interval == 0:
			env_kwargs_eval = dict(
				obs=ObservationType.KIN,
				act=ActionType.RPM,
				initial_xyzs=INIT_XYZS,
				initial_rpys=INIT_RPYS,
				pyb_freq=args.pyb_freq,
				ctrl_freq=args.ctrl_freq,
			)

			avg_return, q_mean, actor_sat_pct, monitor = evaluate_policy_with_gui_monitor(
				env_kwargs=env_kwargs_eval,
				agent=agent,
				preprocess_state_fn=_preprocess_state,
				replay_buffer=buffer,
				episodes=args.eval_episodes,
				gui=args.eval_gui,
				eval_stepsleep=args.eval_stepsleep,
				print_debug=False,
				train_steps_this_tick=train_steps_this_tick,
				q_batch_size=args.eval_q_batch_size,
			)

			writer.add_scalar("Reward/Eval_AverageReturn", avg_return, t)

			# Handle N/A cases
			q_mean_str = "N/A" if q_mean is None or (isinstance(q_mean, float) and np.isnan(q_mean)) else f"{q_mean:.3f}"
			actor_sat_pct_str = "N/A" if actor_sat_pct is None or (isinstance(actor_sat_pct, float) and np.isnan(actor_sat_pct)) else f"{actor_sat_pct:.3f}"
			eval_ret_str = f"{avg_return:.3f}" if avg_return is not None else "N/A"

			rpms_list = [f"{x:.1f}" for x in monitor["motor_rpms"].tolist()]
			sat_list = [f"{x:.1f}" for x in monitor["sat_pct"].tolist()]
			r_deg, p_deg, y_deg = monitor["rpy_deg"]

			print(f"====================== Physics Monitor @ STEP {t} ======================")
			print(f"EvalRet: {eval_ret_str} | Q mean: {q_mean_str} | dyn_loss: N/A")
			print(f"Position: x={monitor['pos'][0]:+.2f}, y={monitor['pos'][1]:+.2f}, z={monitor['pos'][2]:+.2f}")
			print(f"Attitude(deg): R={r_deg:+.1f}, P={p_deg:+.1f}, Y={y_deg:+.1f}")
			print(f"Z velocity: {monitor['z_vel']:+.3f} | Target distance: {monitor['dist']:+.2f}")
			print(f"Motor RPMs: [{', '.join(rpms_list)}] | sat%: [{', '.join(sat_list)}]%")
			print(f"Front-back motor speed difference: {monitor['motor_front_back_diff']:.0f} RPM")
			print(
				f"Action saturation: {monitor['action_sat_pct']:.1f}% | train_steps_this_tick: {train_steps_this_tick} | "
				f"latent_norm: N/A | avg_thrust: {monitor['avg_thrust']:.3f} | hover_loss: N/A | actor_sat_pct(train): {actor_sat_pct_str}"
			)
			print("========================================================================")

			# Reset training step counter for next period
			train_steps_this_tick = 0

			if avg_return > best_eval_return:
				best_eval_return = avg_return
				agent.save(os.path.join(log_dir, "best"))
				print(f"[INFO] New best eval return: {best_eval_return:.3f}. Saved to {log_dir}/best_*.pt")

	env.close()
	writer.close()
	print(f"[INFO] Training finished. best_eval_return={best_eval_return:.3f}")


def _make_argparser():
	parser = argparse.ArgumentParser(description="TD3 training for UAV hovering (HoverAviary).")
	parser.add_argument("--seed", type=int, default=42)
	def str2bool(v):
		if isinstance(v, bool):
			return v
		v_str = str(v).strip().lower()
		if v_str in ("1", "true", "t", "yes", "y"):
			return True
		if v_str in ("0", "false", "f", "no", "n"):
			return False
		raise argparse.ArgumentTypeError(f"Invalid boolean value: {v!r}")

	parser.add_argument("--gui", type=str2bool, default=False)

	# HoverAviary timing (affects state_dim because obs includes action buffer)
	parser.add_argument("--pyb_freq", type=int, default=240)
	parser.add_argument("--ctrl_freq", type=int, default=120)

	# Replay buffer / optimization
	parser.add_argument("--buffer_size", type=int, default=1_000_000)
	parser.add_argument("--batch_size", type=int, default=256)
	parser.add_argument("--total_steps", type=int, default=500_000)
	parser.add_argument("--start_timesteps", type=int, default=10_000)
	parser.add_argument("--update_after", type=int, default=10_000)
	parser.add_argument("--train_every", type=int, default=1)

	# TD3 hyper-params
	parser.add_argument("--gamma", type=float, default=0.99)
	parser.add_argument("--tau", type=float, default=0.005)
	parser.add_argument("--policy_noise", type=float, default=0.2)
	parser.add_argument("--noise_clip", type=float, default=0.5)
	parser.add_argument("--policy_freq", type=int, default=2)
	parser.add_argument("--actor_updates_encoder", type=str2bool, default=False)
	parser.add_argument("--latent_input_scale", type=float, default=0.1)
	parser.add_argument("--grad_clip_norm", type=float, default=1.0)
	parser.add_argument("--reward_scale", type=float, default=0.01)

	# Exploration noise schedule
	parser.add_argument("--expl_noise_start", type=float, default=0.5)
	parser.add_argument("--expl_noise_end", type=float, default=0.1)
	parser.add_argument("--noise_decay_steps", type=float, default=100_000)

	# Evaluation
	parser.add_argument("--eval_interval", type=int, default=10_000)
	parser.add_argument("--eval_episodes", type=int, default=1)
	parser.add_argument("--eval_gui", type=str2bool, default=True)
	parser.add_argument("--eval_stepsleep", type=str2bool, default=True)
	parser.add_argument("--eval_q_batch_size", type=int, default=256)

	parser.add_argument("--output_dir", type=str, default="runs")
	return parser


if __name__ == "__main__":
	args = _make_argparser().parse_args()
	train(args)
