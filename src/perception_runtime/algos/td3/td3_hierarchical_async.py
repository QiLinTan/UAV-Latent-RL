from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from algos.td3.networks import Critic, ResidualMotorActor
from algos.td3.td3_latent_only import _grad_norm
from envs.observation_layout import KIN_DIM, ForestObservationLayout
from models.encoder import Encoder
from models.heads import DynHead, ReconHead
from models.hierarchical_async import AsyncReferenceCache, ReferenceSequenceHead


class TD3HierarchicalAsync:
    """Two-timescale TD3 with cached high-level references and RPM-level actions."""

    _RAY_DIRECTIONS = (
        (1.0, 0.0),
        (0.7071, 0.7071),
        (0.0, 1.0),
        (-0.7071, 0.7071),
        (-1.0, 0.0),
        (-0.7071, -0.7071),
        (0.0, -1.0),
        (0.7071, -0.7071),
    )

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
        latent_dim=16,
        latent_input_scale=1.0,
        grad_clip_norm=1.0,
        sequence_length=15,
        high_level_interval=8,
        reference_valid_steps=120,
        reference_loss_weight=0.05,
        reference_smoothness_weight=0.01,
        high_level_actor_grad_scale=0.0,
        action_l2_weight=0.10,
        motor_balance_weight=0.10,
        action_delta_weight=0.10,
        normalize_actor_q=True,
        motor_collective_fraction=0.60,
        motor_differential_fraction=0.25,
        lookahead_points=3,
        device=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.max_action = float(max_action)
        self.discount = float(discount)
        self.tau = float(tau)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.policy_freq = int(policy_freq)
        self.latent_dim = int(latent_dim)
        self.latent_input_scale = float(latent_input_scale)
        self.grad_clip_norm = float(grad_clip_norm)
        self.sequence_length = int(sequence_length)
        self.high_level_interval = int(high_level_interval)
        self.reference_valid_steps = int(reference_valid_steps)
        self.reference_loss_weight = float(reference_loss_weight)
        self.reference_smoothness_weight = float(reference_smoothness_weight)
        self.high_level_actor_grad_scale = float(high_level_actor_grad_scale)
        self.action_l2_weight = float(action_l2_weight)
        self.motor_balance_weight = float(motor_balance_weight)
        self.action_delta_weight = float(action_delta_weight)
        self.normalize_actor_q = bool(normalize_actor_q)
        self.motor_collective_fraction = float(motor_collective_fraction)
        self.motor_differential_fraction = float(motor_differential_fraction)
        self.lookahead_points = int(lookahead_points)
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2.")
        if self.high_level_interval <= 0:
            raise ValueError("high_level_interval must be positive.")
        if self.reference_valid_steps < self.high_level_interval:
            raise ValueError("reference_valid_steps must be at least high_level_interval.")

        self.layout = ForestObservationLayout.from_total_dim(self.state_dim, action_dim=self.action_dim)
        self.encoder = Encoder(self.state_dim, latent_dim=self.latent_dim).to(self.device)
        self.reference_head = ReferenceSequenceHead(
            self.latent_dim,
            sequence_length=self.sequence_length,
            reference_dim=3,
        ).to(self.device)
        self.recon_head = ReconHead(latent_dim=self.latent_dim, state_dim=KIN_DIM).to(self.device)
        self.dyn_head = DynHead(
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            state_dim=KIN_DIM,
        ).to(self.device)

        self.policy_input_dim = KIN_DIM + self.latent_dim + 3 + 3 + 1 + 1 + self.action_dim
        self.context_dim = self.policy_input_dim
        self.uses_context_replay = True
        self.context_previous_action_start = self.policy_input_dim - self.action_dim
        self.actor = ResidualMotorActor(
            self.policy_input_dim,
            self.action_dim,
            self.max_action,
            collective_fraction=self.motor_collective_fraction,
            differential_fraction=self.motor_differential_fraction,
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(self.policy_input_dim, self.action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=3e-4)
        self.reference_optimizer = torch.optim.Adam(self.reference_head.parameters(), lr=3e-4)
        self.recon_optimizer = torch.optim.Adam(self.recon_head.parameters(), lr=1e-3)
        self.dyn_optimizer = torch.optim.Adam(self.dyn_head.parameters(), lr=1e-3)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.reference_cache = AsyncReferenceCache()
        self.high_level_enabled = True
        self.previous_action = np.zeros(self.action_dim, dtype=np.float32)
        self.runtime_step = 0
        self.total_it = 0
        self.env_step = 0
        self.last_runtime_info = {}

    def _encode(self, state: torch.Tensor) -> torch.Tensor:
        # A bounded high-level interface prevents the low-level actor from
        # amplifying representation drift over long off-policy runs.
        return torch.tanh(self.encoder(state))

    def set_env_step(self, step: int):
        self.env_step = int(step)

    def reset_episode(self):
        self.reference_cache.clear()
        self.previous_action.fill(0.0)
        self.runtime_step = 0
        self.last_runtime_info = {}

    def set_high_level_enabled(self, enabled: bool):
        """Enable/disable high-level refreshes to emulate delay or communication loss."""
        self.high_level_enabled = bool(enabled)

    def _target_reference_sequence(self, state: torch.Tensor) -> torch.Tensor:
        """Build a weak, obstacle-aware local reference target from goal/range observations."""
        parts = self.layout.split(state)
        goal = parts.goal
        ranges = parts.ranges.clamp(0.0, 1.0)
        goal_xy = goal[:, :2]
        goal_norm = goal_xy.norm(dim=1, keepdim=True).clamp_min(1e-6)
        goal_direction = goal_xy / goal_norm

        directions = state.new_tensor(self._RAY_DIRECTIONS)
        repulsion_weights = (1.0 - ranges).pow(2)
        repulsion = -(repulsion_weights.unsqueeze(-1) * directions.unsqueeze(0)).sum(dim=1)
        repulsion = repulsion / repulsion.norm(dim=1, keepdim=True).clamp_min(1.0)
        local_direction = goal_direction + 0.45 * repulsion
        local_direction = local_direction / local_direction.norm(dim=1, keepdim=True).clamp_min(1e-6)

        fractions = torch.linspace(
            1.0 / self.sequence_length,
            1.0,
            self.sequence_length,
            device=state.device,
            dtype=state.dtype,
        ).view(1, -1, 1)
        planar_distance = goal_norm.clamp(max=1.0).unsqueeze(1)
        xy = fractions * planar_distance * local_direction.unsqueeze(1)
        target_xy = fractions * goal_xy.unsqueeze(1)
        blend = fractions.pow(1.5)
        xy = (1.0 - blend) * xy + blend * target_xy
        z = fractions * goal[:, 2:3].unsqueeze(1)
        return torch.cat([xy, z], dim=2).clamp(-1.0, 1.0)

    def _reference_loss(self, sequence: torch.Tensor, state: torch.Tensor):
        target = self._target_reference_sequence(state)
        tracking_loss = F.smooth_l1_loss(sequence, target)
        velocity = sequence[:, 1:] - sequence[:, :-1]
        acceleration = velocity[:, 1:] - velocity[:, :-1]
        smoothness_loss = acceleration.pow(2).mean()
        return tracking_loss, smoothness_loss, target

    def _refresh_runtime_cache(self, state: torch.Tensor):
        latent = self._encode(state)
        sequence = self.reference_head(latent)
        snapshot = self.reference_cache.update(
            sequence[0].detach().cpu().numpy(),
            latent[0].detach().cpu().numpy(),
            created_step=self.runtime_step,
            point_interval=self.high_level_interval,
            valid_steps=self.reference_valid_steps,
        )
        return snapshot

    def prepare_runtime_context(self, state) -> np.ndarray:
        """Build the exact context used by the next low-level action."""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        with torch.no_grad():
            must_refresh = (
                not self.reference_cache.has_value
                or (
                    self.runtime_step % self.high_level_interval == 0
                    and self.reference_cache.created_step != self.runtime_step
                )
            )
            if must_refresh and self.high_level_enabled:
                self._refresh_runtime_cache(state_t)

            if not self.reference_cache.has_value:
                context = np.concatenate(
                    [
                        state_t[0, :KIN_DIM].cpu().numpy(),
                        np.zeros(self.latent_dim + 3 + 3, dtype=np.float32),
                        np.array([1.0, 0.0], dtype=np.float32),
                        self.previous_action,
                    ]
                ).astype(np.float32)
                self.last_runtime_info = {
                    "async_cache_version": 0.0,
                    "async_cache_age_steps": 0.0,
                    "async_cache_age_ratio": 1.0,
                    "async_cache_valid": 0.0,
                    "async_reference_index": 0.0,
                    "async_high_level_refreshed": 0.0,
                    "async_safe_fallback": 1.0,
                }
                return context

            cached = self.reference_cache.sample(
                self.runtime_step,
                lookahead_points=self.lookahead_points,
            )
            if not cached["valid"]:
                context = np.concatenate(
                    [
                        state_t[0, :KIN_DIM].cpu().numpy(),
                        cached["latent"] * self.latent_input_scale,
                        cached["current"],
                        cached["lookahead"],
                        np.array([cached["age_ratio"], 0.0], dtype=np.float32),
                        self.previous_action,
                    ]
                ).astype(np.float32)
                self.last_runtime_info = {
                    "async_cache_version": float(cached["version"]),
                    "async_cache_age_steps": float(cached["age_steps"]),
                    "async_cache_age_ratio": float(cached["age_ratio"]),
                    "async_cache_valid": 0.0,
                    "async_reference_index": float(cached["lower_index"]),
                    "async_high_level_refreshed": 0.0,
                    "async_safe_fallback": 1.0,
                }
                return context

            context = np.concatenate(
                [
                    state_t[0, :KIN_DIM].cpu().numpy(),
                    cached["latent"] * self.latent_input_scale,
                    cached["current"],
                    cached["lookahead"],
                    np.array([cached["age_ratio"], float(cached["valid"])], dtype=np.float32),
                    self.previous_action,
                ]
            ).astype(np.float32)

        self.last_runtime_info = {
            "async_cache_version": float(cached["version"]),
            "async_cache_age_steps": float(cached["age_steps"]),
            "async_cache_age_ratio": float(cached["age_ratio"]),
            "async_cache_valid": float(cached["valid"]),
            "async_reference_index": float(cached["lower_index"]),
            "async_high_level_refreshed": float(must_refresh and self.high_level_enabled),
            "async_safe_fallback": 0.0,
        }
        return context

    def action_from_context(self, context) -> np.ndarray:
        if not bool(self.last_runtime_info.get("async_cache_valid", 0.0)):
            return np.zeros(self.action_dim, dtype=np.float32)
        context_t = torch.as_tensor(
            context,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, -1)
        with torch.no_grad():
            action = self.actor(context_t)
        return action.cpu().numpy().flatten()

    def project_motor_action(self, action) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        collective_limit = self.max_action * self.motor_collective_fraction
        differential_limit = self.max_action * self.motor_differential_fraction
        collective = float(np.clip(action.mean(), -collective_limit, collective_limit))
        differential = action - float(action.mean())
        differential -= float(differential.mean())
        max_abs_differential = float(np.abs(differential).max())
        if max_abs_differential > differential_limit:
            differential *= differential_limit / max(max_abs_differential, 1e-6)
        return np.clip(
            collective + differential,
            -self.max_action,
            self.max_action,
        ).astype(np.float32)

    def sample_safe_random_action(self) -> np.ndarray:
        collective_limit = self.max_action * self.motor_collective_fraction
        differential_limit = self.max_action * self.motor_differential_fraction
        collective = np.random.uniform(-0.5 * collective_limit, 0.5 * collective_limit)
        differential = np.random.uniform(
            -0.5 * differential_limit,
            0.5 * differential_limit,
            size=self.action_dim,
        )
        return self.project_motor_action(collective + differential)

    def add_safe_exploration_noise(self, action, noise_std: float) -> np.ndarray:
        collective_noise = 0.5 * float(noise_std) * np.random.randn()
        differential_noise = 0.5 * float(noise_std) * np.random.randn(self.action_dim)
        differential_noise -= differential_noise.mean()
        return self.project_motor_action(
            np.asarray(action, dtype=np.float32) + collective_noise + differential_noise
        )

    def record_executed_action(self, action):
        self.previous_action = np.asarray(action, dtype=np.float32).reshape(self.action_dim).copy()

    def advance_runtime_step(self):
        self.runtime_step += 1

    def select_action(self, state):
        context = self.prepare_runtime_context(state)
        action = self.action_from_context(context)
        self.record_executed_action(action)
        self.advance_runtime_step()
        return action

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        (
            state,
            action,
            next_state,
            reward,
            not_done,
            context,
            next_context,
        ) = replay_buffer.sample(batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.to(self.device)
        not_done = not_done.to(self.device)
        context = context.to(self.device)
        next_context = next_context.to(self.device)

        latent_rep = self._encode(state)
        sequence_rep = self.reference_head(latent_rep)
        recon_loss = F.mse_loss(self.recon_head(latent_rep), state[:, :KIN_DIM])
        dyn_loss = F.mse_loss(self.dyn_head(latent_rep, action), next_state[:, :KIN_DIM])
        reference_loss, reference_smoothness_loss, reference_target = self._reference_loss(
            sequence_rep,
            state,
        )
        representation_loss = 0.1 * (recon_loss + dyn_loss)
        reference_weighted_loss = self.reference_loss_weight * reference_loss
        smoothness_weighted_loss = self.reference_smoothness_weight * reference_smoothness_loss

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip,
                self.noise_clip,
            )
            next_action = (self.actor_target(next_context) + noise).clamp(
                -self.max_action,
                self.max_action,
            )
            target_q1, target_q2 = self.critic_target(next_context, next_action)
            target_q = reward + not_done * self.discount * torch.min(target_q1, target_q2)

        current_q1, current_q2 = self.critic(context, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        total_loss = (
            critic_loss
            + representation_loss
            + reference_weighted_loss
            + smoothness_weighted_loss
        )

        self.encoder_optimizer.zero_grad()
        self.reference_optimizer.zero_grad()
        self.recon_optimizer.zero_grad()
        self.dyn_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        encoder_grad_norm_rep = _grad_norm(self.encoder.parameters())
        reference_grad_norm_supervised = _grad_norm(self.reference_head.parameters())
        critic_grad_norm = _grad_norm(self.critic.parameters())
        for module in (
            self.encoder,
            self.reference_head,
            self.recon_head,
            self.dyn_head,
            self.critic,
        ):
            torch.nn.utils.clip_grad_norm_(module.parameters(), self.grad_clip_norm)
        self.encoder_optimizer.step()
        self.reference_optimizer.step()
        self.recon_optimizer.step()
        self.dyn_optimizer.step()
        self.critic_optimizer.step()

        actor_loss_val = None
        actor_grad_norm = 0.0
        encoder_grad_norm_actor = 0.0
        reference_grad_norm_actor = 0.0
        actor_sat_pct = None
        actor_updated = False
        if self.total_it % self.policy_freq == 0:
            actor_actions = self.actor(context)
            previous_action = context[:, self.context_previous_action_start :]
            action_l2_loss = actor_actions.pow(2).mean()
            motor_balance_loss = actor_actions.var(dim=1, unbiased=False).mean()
            action_delta_loss = (actor_actions - previous_action).pow(2).mean()
            actor_q = self.critic.Q1(context, actor_actions)
            if self.normalize_actor_q:
                actor_q_scale = actor_q.detach().abs().mean().clamp_min(1.0)
            else:
                actor_q_scale = actor_q.new_tensor(1.0)
            actor_loss = (
                -actor_q.mean() / actor_q_scale
                + self.action_l2_weight * action_l2_loss
                + self.motor_balance_weight * motor_balance_loss
                + self.action_delta_weight * action_delta_loss
            )

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = _grad_norm(self.actor.parameters())
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
            self.actor_optimizer.step()

            actor_loss_val = float(actor_loss.item())
            actor_sat_pct = float(
                (actor_actions.detach().abs() >= (self.max_action - 1e-3)).float().mean().item()
            )
            actor_updated = True

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        with torch.no_grad():
            endpoint_error = (sequence_rep[:, -1] - reference_target[:, -1]).norm(dim=1).mean()
            reference_step_size = (sequence_rep[:, 1:] - sequence_rep[:, :-1]).norm(dim=2).mean()
            latent_stats = self._encode(state)

        return {
            "critic_loss": float(critic_loss.item()),
            "critic_grad_norm": float(critic_grad_norm),
            "recon_loss": float(recon_loss.item()),
            "dyn_loss": float(dyn_loss.item()),
            "reference_loss": float(reference_loss.item()),
            "reference_weighted_loss": float(reference_weighted_loss.detach().item()),
            "reference_smoothness_loss": float(reference_smoothness_loss.item()),
            "reference_smoothness_weighted_loss": float(smoothness_weighted_loss.detach().item()),
            "reference_endpoint_error": float(endpoint_error.item()),
            "reference_step_size": float(reference_step_size.item()),
            "reference_grad_norm_supervised": float(reference_grad_norm_supervised),
            "reference_grad_norm_actor": float(reference_grad_norm_actor),
            "encoder_grad_norm_rep": float(encoder_grad_norm_rep),
            "encoder_grad_norm_actor": float(encoder_grad_norm_actor),
            "actor_loss": actor_loss_val,
            "action_l2_loss": (
                float(action_l2_loss.detach().item()) if actor_updated else None
            ),
            "motor_balance_loss": (
                float(motor_balance_loss.detach().item()) if actor_updated else None
            ),
            "action_delta_loss": (
                float(action_delta_loss.detach().item()) if actor_updated else None
            ),
            "actor_q_scale": (
                float(actor_q_scale.detach().item()) if actor_updated else None
            ),
            "actor_grad_norm": float(actor_grad_norm),
            "actor_updated": actor_updated,
            "actor_sat_pct": actor_sat_pct,
            "latent_std_mean": float(latent_stats.std(dim=0).mean().item()),
            "latent_abs_mean": float(latent_stats.abs().mean().item()),
            "q1_mean": float(current_q1.detach().mean().item()),
            "q2_mean": float(current_q2.detach().mean().item()),
            "q_target_mean": float(target_q.detach().mean().item()),
            "q_gap_abs_mean": float((current_q1.detach() - current_q2.detach()).abs().mean().item()),
            "hierarchical_async": 1.0,
            "reference_sequence_length": float(self.sequence_length),
            "high_level_interval": float(self.high_level_interval),
            "reference_valid_steps": float(self.reference_valid_steps),
            "reference_loss_weight": float(self.reference_loss_weight),
            "high_level_actor_grad_scale": float(self.high_level_actor_grad_scale),
            "action_l2_weight": float(self.action_l2_weight),
            "motor_balance_weight": float(self.motor_balance_weight),
            "action_delta_weight": float(self.action_delta_weight),
            "normalize_actor_q": float(self.normalize_actor_q),
            "context_replay_enabled": 1.0,
            "motor_collective_fraction": float(self.motor_collective_fraction),
            "motor_differential_fraction": float(self.motor_differential_fraction),
            "env_step": float(self.env_step),
            **self.last_runtime_info,
        }

    def save(self, filename):
        modules = {
            "critic": self.critic,
            "encoder": self.encoder,
            "reference_head": self.reference_head,
            "recon_head": self.recon_head,
            "dyn_head": self.dyn_head,
            "actor": self.actor,
        }
        optimizers = {
            "critic_optimizer": self.critic_optimizer,
            "encoder_optimizer": self.encoder_optimizer,
            "reference_optimizer": self.reference_optimizer,
            "recon_optimizer": self.recon_optimizer,
            "dyn_optimizer": self.dyn_optimizer,
            "actor_optimizer": self.actor_optimizer,
        }
        for suffix, module in modules.items():
            torch.save(module.state_dict(), f"{filename}_{suffix}")
        for suffix, optimizer in optimizers.items():
            torch.save(optimizer.state_dict(), f"{filename}_{suffix}")

    def load(self, filename):
        modules = {
            "critic": self.critic,
            "encoder": self.encoder,
            "reference_head": self.reference_head,
            "recon_head": self.recon_head,
            "dyn_head": self.dyn_head,
            "actor": self.actor,
        }
        optimizers = {
            "critic_optimizer": self.critic_optimizer,
            "encoder_optimizer": self.encoder_optimizer,
            "reference_optimizer": self.reference_optimizer,
            "recon_optimizer": self.recon_optimizer,
            "dyn_optimizer": self.dyn_optimizer,
            "actor_optimizer": self.actor_optimizer,
        }
        for suffix, module in modules.items():
            module.load_state_dict(torch.load(f"{filename}_{suffix}", map_location=self.device))
        for suffix, optimizer in optimizers.items():
            optimizer.load_state_dict(torch.load(f"{filename}_{suffix}", map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
        self.reset_episode()
