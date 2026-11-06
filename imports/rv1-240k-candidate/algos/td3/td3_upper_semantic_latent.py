from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from algos.td3.networks import Actor, Critic
from algos.td3.td3_plain import _grad_norm
from models.encoder import Encoder


class SemanticResidualActor(nn.Module):
    """Frozen proven policy plus a bounded history-conditioned latent residual."""

    def __init__(self, state_dim, action_dim, max_action, base_state_dim=29, latent_dim=16, residual_scale=0.25):
        super().__init__()
        self.base_state_dim = int(base_state_dim)
        self.max_action = float(max_action)
        self.residual_scale = float(residual_scale)
        self.base_actor = Actor(self.base_state_dim, action_dim, max_action)
        for parameter in self.base_actor.parameters():
            parameter.requires_grad_(False)
        self.encoder = Encoder(state_dim, latent_dim=latent_dim)
        self.residual = nn.Sequential(
            nn.Linear(state_dim + latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, state):
        with torch.no_grad():
            base_action = self.base_actor(state[:, : self.base_state_dim])
        latent = self.encoder(state)
        correction = self.residual_scale * torch.tanh(
            self.residual(torch.cat([state, latent], dim=1))
        )
        return (base_action + correction).clamp(-self.max_action, self.max_action)


class TD3UpperSemanticLatent:
    """Upper TD3 with temporal perception latent and interpretable affordance heads."""

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        *,
        base_policy_checkpoint,
        base_state_dim=29,
        latent_dim=16,
        residual_scale=0.25,
        semantic_loss_weight=0.2,
        danger_range=0.20,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        grad_clip_norm=1.0,
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
        self.semantic_loss_weight = float(semantic_loss_weight)
        self.danger_range = float(danger_range)
        self.range_slice = slice(15, 23)
        self.goal_slice = slice(12, 15)

        self.actor = SemanticResidualActor(
            state_dim, action_dim, max_action,
            base_state_dim=base_state_dim,
            latent_dim=latent_dim,
            residual_scale=residual_scale,
        ).to(self.device)
        base_state = torch.load(base_policy_checkpoint + "_actor", map_location=self.device)
        self.actor.base_actor.load_state_dict(base_state)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.clearance_head = nn.Linear(latent_dim, 8).to(self.device)
        self.risk_head = nn.Linear(latent_dim, 1).to(self.device)
        self.progress_head = nn.Linear(latent_dim, 1).to(self.device)
        trainable_actor = [p for p in self.actor.parameters() if p.requires_grad]
        self.actor_optimizer = torch.optim.Adam(trainable_actor, lr=3e-4)
        self.semantic_head_optimizer = torch.optim.Adam(
            list(self.clearance_head.parameters())
            + list(self.risk_head.parameters())
            + list(self.progress_head.parameters()),
            lr=3e-4,
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.total_it = 0

    def select_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        with torch.no_grad():
            return self.actor(state).cpu().numpy().flatten()

    def _semantic_loss(self, state, next_state):
        latent = self.actor.encoder(state)
        next_ranges = next_state[:, self.range_slice].clamp(0.0, 1.0)
        clearance_loss = F.mse_loss(torch.sigmoid(self.clearance_head(latent)), next_ranges)
        risk_target = (next_ranges.amin(dim=1, keepdim=True) < self.danger_range).float()
        risk_loss = F.binary_cross_entropy_with_logits(self.risk_head(latent), risk_target)
        goal_distance = torch.linalg.vector_norm(state[:, self.goal_slice], dim=1, keepdim=True)
        next_goal_distance = torch.linalg.vector_norm(next_state[:, self.goal_slice], dim=1, keepdim=True)
        progress_target = (goal_distance - next_goal_distance).clamp(-0.25, 0.25)
        progress_loss = F.mse_loss(self.progress_head(latent), progress_target)
        return clearance_loss, risk_loss, progress_loss

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        state, action, next_state = state.to(self.device), action.to(self.device), next_state.to(self.device)
        reward, not_done = reward.to(self.device), not_done.to(self.device)

        clearance_loss, risk_loss, progress_loss = self._semantic_loss(state, next_state)
        semantic_loss = clearance_loss + risk_loss + progress_loss
        self.actor_optimizer.zero_grad()
        self.semantic_head_optimizer.zero_grad()
        (self.semantic_loss_weight * semantic_loss).backward()
        semantic_grad_norm = _grad_norm(self.actor.encoder.parameters())
        torch.nn.utils.clip_grad_norm_(self.actor.encoder.parameters(), self.grad_clip_norm)
        self.actor_optimizer.step()
        self.semantic_head_optimizer.step()

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = reward + not_done * self.discount * torch.min(target_q1, target_q2)
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
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = _grad_norm(p for p in self.actor.parameters() if p.requires_grad)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.actor.parameters() if p.requires_grad], self.grad_clip_norm
            )
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.item())
            actor_updated = True
            with torch.no_grad():
                actor_actions = self.actor(state)
                actor_sat_pct = float((actor_actions.abs() >= self.max_action - 1e-3).float().mean())
            for parameter, target in zip(self.actor.parameters(), self.actor_target.parameters()):
                target.data.copy_(self.tau * parameter.data + (1.0 - self.tau) * target.data)
            for parameter, target in zip(self.critic.parameters(), self.critic_target.parameters()):
                target.data.copy_(self.tau * parameter.data + (1.0 - self.tau) * target.data)

        with torch.no_grad():
            latent = self.actor.encoder(state)
        return {
            "critic_loss": float(critic_loss.item()),
            "critic_grad_norm": float(critic_grad_norm),
            "actor_loss": actor_loss_value,
            "actor_grad_norm": float(actor_grad_norm),
            "actor_updated": actor_updated,
            "actor_sat_pct": actor_sat_pct,
            "semantic_loss": float(semantic_loss.item()),
            "clearance_loss": float(clearance_loss.item()),
            "risk_loss": float(risk_loss.item()),
            "progress_loss": float(progress_loss.item()),
            "semantic_grad_norm": float(semantic_grad_norm),
            "latent_abs_mean": float(latent.abs().mean().item()),
            "latent_std_mean": float(latent.std(dim=0).mean().item()),
        }

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(
            {
                "clearance_head": self.clearance_head.state_dict(),
                "risk_head": self.risk_head.state_dict(),
                "progress_head": self.progress_head.state_dict(),
                "optimizer": self.semantic_head_optimizer.state_dict(),
            },
            filename + "_semantic_heads",
        )

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=self.device))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic)
        heads = torch.load(filename + "_semantic_heads", map_location=self.device)
        self.clearance_head.load_state_dict(heads["clearance_head"])
        self.risk_head.load_state_dict(heads["risk_head"])
        self.progress_head.load_state_dict(heads["progress_head"])
        self.semantic_head_optimizer.load_state_dict(heads["optimizer"])
