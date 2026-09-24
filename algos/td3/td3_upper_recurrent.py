from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algos.td3.td3_plain import _grad_norm


def _project_planar_action_tensor(action: torch.Tensor, max_action: float) -> torch.Tensor:
    """Project XY onto the executable disk while preserving the z component."""

    if action.shape[-1] < 2:
        return action
    planar = action[..., :2]
    norm = torch.linalg.vector_norm(planar, dim=-1, keepdim=True)
    scale = torch.clamp(float(max_action) / torch.clamp(norm, min=1e-8), max=1.0)
    return torch.cat((planar * scale, action[..., 2:]), dim=-1)


def _squash_planar_logits_to_disk(logits: torch.Tensor, max_action: float) -> torch.Tensor:
    """Smoothly map unconstrained XY logits into the executable action disk."""

    planar_logits = logits[..., :2]
    norm = torch.linalg.vector_norm(planar_logits, dim=-1, keepdim=True)
    direction = planar_logits / torch.clamp(norm, min=1e-8)
    planar = float(max_action) * torch.tanh(norm) * direction
    planar = torch.where(norm > 1e-8, planar, torch.zeros_like(planar))
    remainder = float(max_action) * torch.tanh(logits[..., 2:])
    return torch.cat((planar, remainder), dim=-1)


class _RecurrentStateEncoder(nn.Module):
    """Encode the explicit upper-policy history while retaining current state."""

    def __init__(
        self,
        state_dim: int,
        *,
        base_state_dim: int,
        history_length: int,
        history_frame_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.base_state_dim = int(base_state_dim)
        self.history_length = int(history_length)
        self.history_frame_dim = int(history_frame_dim)
        self.hidden_dim = int(hidden_dim)
        expected_dim = self.base_state_dim + self.history_length * self.history_frame_dim
        if self.history_length <= 0:
            raise ValueError("history_length must be positive for recurrent upper TD3.")
        if self.history_frame_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("history_frame_dim and hidden_dim must be positive.")
        if self.state_dim != expected_dim:
            raise ValueError(
                "Recurrent upper-state contract mismatch: "
                f"state_dim={self.state_dim}, expected {expected_dim} "
                f"({self.base_state_dim} + {self.history_length}*{self.history_frame_dim})."
            )
        self.gru = nn.GRU(
            input_size=self.history_frame_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )

    @property
    def output_dim(self) -> int:
        return self.base_state_dim + self.hidden_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(
                f"state must have shape [batch, {self.state_dim}], got {tuple(state.shape)}."
            )
        current = state[:, : self.base_state_dim]
        history = state[:, self.base_state_dim :].reshape(
            -1, self.history_length, self.history_frame_dim
        )
        _, hidden = self.gru(history)
        return torch.cat((current, hidden[-1]), dim=1)


class RecurrentUpperActor(nn.Module):
    """GRU upper actor producing a bounded 3-D reference-velocity command."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        *,
        base_state_dim: int = 29,
        history_length: int = 8,
        history_frame_dim: int = 12,
        hidden_dim: int = 32,
        project_planar_action: bool = False,
    ):
        super().__init__()
        self.max_action = float(max_action)
        self.project_planar_action = bool(project_planar_action)
        self.encoder = _RecurrentStateEncoder(
            state_dim,
            base_state_dim=base_state_dim,
            history_length=history_length,
            history_frame_dim=history_frame_dim,
            hidden_dim=hidden_dim,
        )
        self.policy = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, int(action_dim)),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.policy(self.encoder(state))
        if self.project_planar_action:
            return _squash_planar_logits_to_disk(logits, self.max_action)
        return self.max_action * torch.tanh(logits)


class _RecurrentQBranch(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        base_state_dim: int,
        history_length: int,
        history_frame_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.encoder = _RecurrentStateEncoder(
            state_dim,
            base_state_dim=base_state_dim,
            history_length=history_length,
            history_frame_dim=history_frame_dim,
            hidden_dim=hidden_dim,
        )
        self.value = nn.Sequential(
            nn.Linear(self.encoder.output_dim + int(action_dim), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        feature = self.encoder(state)
        return self.value(torch.cat((feature, action), dim=1))


class RecurrentTwinCritic(nn.Module):
    """Two independent recurrent Q estimators used by upper-level TD3."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        base_state_dim: int = 29,
        history_length: int = 8,
        history_frame_dim: int = 12,
        hidden_dim: int = 32,
    ):
        super().__init__()
        branch_kwargs = dict(
            base_state_dim=base_state_dim,
            history_length=history_length,
            history_frame_dim=history_frame_dim,
            hidden_dim=hidden_dim,
        )
        self.q1 = _RecurrentQBranch(state_dim, action_dim, **branch_kwargs)
        self.q2 = _RecurrentQBranch(state_dim, action_dim, **branch_kwargs)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        return self.q1(state, action), self.q2(state, action)

    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(state, action)


class TD3UpperRecurrent:
    """Paper-inspired recurrent double-critic TD3 for ``upper_dslpid_*``.

    The policy action remains a normalized 3-D velocity command.  Conversion
    to a world-frame ReferencePacket and all 120 Hz motor control stay inside
    LearnedUpperDSLPIDEnv; this class never emits motor commands.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        *,
        base_state_dim=29,
        history_length=8,
        history_frame_dim=12,
        hidden_dim=32,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        grad_clip_norm=1.0,
        actor_lr=3e-4,
        critic_lr=3e-4,
        conservative_actor_lr=None,
        conservative_critic_lr=None,
        conservative_policy_noise=None,
        conservative_policy_freq=None,
        policy_anchor_weight=0.0,
        project_planar_action=False,
        device=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_action = float(max_action)
        self.discount = float(discount)
        self.tau = float(tau)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.policy_freq = int(policy_freq)
        self.grad_clip_norm = float(grad_clip_norm)
        self.actor_lr = float(actor_lr)
        self.critic_lr = float(critic_lr)
        self.conservative_actor_lr = float(
            self.actor_lr if conservative_actor_lr is None else conservative_actor_lr
        )
        self.conservative_critic_lr = float(
            self.critic_lr if conservative_critic_lr is None else conservative_critic_lr
        )
        self.conservative_policy_noise = float(
            self.policy_noise if conservative_policy_noise is None else conservative_policy_noise
        )
        self.conservative_policy_freq = int(
            self.policy_freq if conservative_policy_freq is None else conservative_policy_freq
        )
        self.policy_anchor_weight = float(policy_anchor_weight)
        self.project_planar_action = bool(project_planar_action)
        if min(
            self.actor_lr,
            self.critic_lr,
            self.conservative_actor_lr,
            self.conservative_critic_lr,
            self.conservative_policy_noise,
            self.policy_anchor_weight,
        ) < 0.0:
            raise ValueError("Learning rates and policy_anchor_weight must be non-negative.")
        if self.conservative_policy_freq <= 0:
            raise ValueError("conservative_policy_freq must be positive.")
        self.base_state_dim = int(base_state_dim)
        self.history_length = int(history_length)
        self.history_frame_dim = int(history_frame_dim)
        self.hidden_dim = int(hidden_dim)

        network_kwargs = dict(
            base_state_dim=self.base_state_dim,
            history_length=self.history_length,
            history_frame_dim=self.history_frame_dim,
            hidden_dim=self.hidden_dim,
        )
        self.actor = RecurrentUpperActor(
            state_dim,
            action_dim,
            self.max_action,
            project_planar_action=self.project_planar_action,
            **network_kwargs,
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = RecurrentTwinCritic(
            state_dim, action_dim, **network_kwargs
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.total_it = 0
        self.uses_success_replay = True
        self.conservative_phase = False
        self._policy_anchor_actor = None

    def select_action(self, state):
        value = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        with torch.no_grad():
            action = self.actor(value)
        return action.cpu().numpy().flatten()

    def project_action(self, action):
        """Apply the same executable-action contract after exploration noise."""

        value = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        value = np.clip(value, -self.max_action, self.max_action)
        if self.project_planar_action and value.size >= 2:
            norm = float(np.linalg.norm(value[:2]))
            if norm > self.max_action:
                value[:2] *= self.max_action / norm
        return value

    def evaluate_critic_actions(self, state, actions):
        """Return both critics' values for candidate actions at one state.

        This is intentionally inference-only and is used by evaluation
        diagnostics; it does not alter recurrent or optimizer state.
        """

        state_value = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        action_values = torch.as_tensor(
            actions, dtype=torch.float32, device=self.device
        )
        if action_values.ndim != 2:
            raise ValueError("actions must have shape [candidate_count, action_dim].")
        state_batch = state_value.expand(action_values.shape[0], -1)
        with torch.no_grad():
            q1, q2 = self.critic(state_batch, action_values)
        return (
            q1.detach().cpu().numpy().reshape(-1),
            q2.detach().cpu().numpy().reshape(-1),
        )

    def enter_conservative_phase(self):
        """Reduce update size after a stable navigation policy is established."""

        if self.conservative_phase:
            return
        for group in self.actor_optimizer.param_groups:
            group["lr"] = self.conservative_actor_lr
        for group in self.critic_optimizer.param_groups:
            group["lr"] = self.conservative_critic_lr
        self.policy_noise = self.conservative_policy_noise
        self.policy_freq = self.conservative_policy_freq
        self.conservative_phase = True

    def set_policy_anchor_from_current(self):
        """Freeze the current actor as a retention target for successful states."""

        if self.policy_anchor_weight <= 0.0:
            return
        self._policy_anchor_actor = copy.deepcopy(self.actor).to(self.device).eval()
        for parameter in self._policy_anchor_actor.parameters():
            parameter.requires_grad_(False)

    def train(
        self,
        replay_buffer,
        batch_size=256,
        success_replay_buffer=None,
        success_replay_fraction=0.0,
    ):
        self.total_it += 1
        success_replay_fraction = float(np.clip(success_replay_fraction, 0.0, 1.0))
        success_batch_size = 0
        if success_replay_buffer is not None and success_replay_buffer.size > 0:
            success_batch_size = min(
                int(batch_size) - 1,
                int(round(int(batch_size) * success_replay_fraction)),
            )
        main_batch_size = int(batch_size) - success_batch_size
        state, action, next_state, reward, not_done = replay_buffer.sample(main_batch_size)
        anchor_states = None
        if success_batch_size > 0:
            success_state, success_action, success_next_state, success_reward, success_not_done = (
                success_replay_buffer.sample(success_batch_size)
            )
            state = torch.cat((state, success_state), dim=0)
            action = torch.cat((action, success_action), dim=0)
            next_state = torch.cat((next_state, success_next_state), dim=0)
            reward = torch.cat((reward, success_reward), dim=0)
            not_done = torch.cat((not_done, success_not_done), dim=0)
            anchor_states = success_state.to(self.device)
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.to(self.device)
        not_done = not_done.to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )
            if self.project_planar_action:
                next_action = _project_planar_action_tensor(next_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = reward + not_done * self.discount * torch.minimum(target_q1, target_q2)

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = _grad_norm(self.critic.parameters())
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()

        actor_loss_value = None
        actor_grad_norm = 0.0
        actor_updated = False
        actor_sat_pct = None
        actor_planar_boundary_pct = None
        actor_policy_anchor_loss = 0.0
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            if self._policy_anchor_actor is not None and anchor_states is not None:
                with torch.no_grad():
                    anchor_action = self._policy_anchor_actor(anchor_states)
                anchor_loss = F.mse_loss(self.actor(anchor_states), anchor_action)
                actor_policy_anchor_loss = float(anchor_loss.item())
                actor_loss = actor_loss + self.policy_anchor_weight * anchor_loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = _grad_norm(self.actor.parameters())
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.item())
            actor_updated = True
            with torch.no_grad():
                actor_action = self.actor(state)
                actor_sat_pct = float(
                    (actor_action.abs() >= self.max_action - 1e-3).float().mean().item()
                )
                actor_planar_boundary_pct = float(
                    (
                        torch.linalg.vector_norm(actor_action[:, :2], dim=1)
                        >= self.max_action - 1e-3
                    )
                    .float()
                    .mean()
                    .item()
                )
            for parameter, target in zip(self.critic.parameters(), self.critic_target.parameters()):
                target.data.copy_(self.tau * parameter.data + (1.0 - self.tau) * target.data)
            for parameter, target in zip(self.actor.parameters(), self.actor_target.parameters()):
                target.data.copy_(self.tau * parameter.data + (1.0 - self.tau) * target.data)

        with torch.no_grad():
            q_gap = (current_q1 - current_q2).abs().mean()
        return {
            "critic_loss": float(critic_loss.item()),
            "critic_grad_norm": float(critic_grad_norm),
            "actor_loss": actor_loss_value,
            "actor_grad_norm": float(actor_grad_norm),
            "actor_updated": actor_updated,
            "actor_sat_pct": actor_sat_pct,
            "actor_planar_boundary_pct": actor_planar_boundary_pct,
            "q1_mean": float(current_q1.mean().item()),
            "q2_mean": float(current_q2.mean().item()),
            "q_target_mean": float(target_q.mean().item()),
            "q_gap_abs_mean": float(q_gap.item()),
            "upper_recurrent": 1.0,
            "upper_history_length": float(self.history_length),
            "upper_recurrent_hidden_dim": float(self.hidden_dim),
            "actor_policy_anchor_loss": float(actor_policy_anchor_loss),
            "success_replay_fraction": float(success_batch_size / max(1, int(batch_size))),
            "conservative_phase": float(self.conservative_phase),
            "actor_lr": float(self.actor_optimizer.param_groups[0]["lr"]),
            "critic_lr": float(self.critic_optimizer.param_groups[0]["lr"]),
            "policy_noise": float(self.policy_noise),
            "policy_freq": float(self.policy_freq),
        }

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer", map_location=self.device)
        )
        self.actor_target = copy.deepcopy(self.actor)
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=self.device))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer", map_location=self.device)
        )
        self.critic_target = copy.deepcopy(self.critic)
