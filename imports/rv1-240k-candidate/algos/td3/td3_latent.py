import copy

import torch
import torch.nn.functional as F

from algos.td3.grad_utils import (
    parse_encoder_grad_schedule,
    resolve_encoder_grad_scale,
    scheduled_encoder_grad_scale,
    soft_detach,
)
from algos.td3.networks import Actor, Critic
from algos.td3.latent_regularizers import posture_separation_loss
from models.encoder import Encoder
from models.heads import DynHead, ReconHead


def _grad_norm(parameters) -> float:
    total_sq = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad.detach()
        total_sq += float(torch.sum(g * g).item())
    return float(total_sq ** 0.5)


class TD3:
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
        actor_updates_encoder=False,
        critic_updates_encoder=False,
        actor_encoder_grad_scale=None,
        critic_encoder_grad_scale=None,
        critic_encoder_grad_schedule=None,
        latent_input_scale=0.1,
        grad_clip_norm=1.0,
        latent_dim=16,
        posture_separation_weight=0.0,
        posture_separation_margin=1.0,
        device=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_action = float(max_action)

        self.latent_dim = int(latent_dim)
        self.latent_input_scale = float(latent_input_scale)
        self.grad_clip_norm = float(grad_clip_norm)
        self.actor_encoder_grad_scale = resolve_encoder_grad_scale(
            actor_encoder_grad_scale,
            actor_updates_encoder,
            default=0.0,
        )
        self.critic_encoder_grad_scale = resolve_encoder_grad_scale(
            critic_encoder_grad_scale,
            critic_updates_encoder,
            default=0.05,
        )
        self.critic_encoder_grad_schedule = parse_encoder_grad_schedule(critic_encoder_grad_schedule)
        self.actor_updates_encoder = self.actor_encoder_grad_scale > 0.0
        self.discount = float(discount)
        self.tau = float(tau)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.policy_freq = int(policy_freq)
        self.critic_updates_encoder = self.critic_encoder_grad_scale > 0.0
        self.posture_separation_weight = float(posture_separation_weight)
        self.posture_separation_margin = float(posture_separation_margin)

        new_state_dim = int(state_dim) + self.latent_dim

        self.encoder = Encoder(state_dim, latent_dim=self.latent_dim).to(self.device)
        self.recon_head = ReconHead(latent_dim=self.latent_dim, state_dim=12).to(self.device)
        self.dyn_head = DynHead(latent_dim=self.latent_dim, action_dim=action_dim, state_dim=12).to(self.device)

        self.actor = Actor(new_state_dim, action_dim, self.max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(new_state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=3e-4)
        self.recon_optimizer = torch.optim.Adam(self.recon_head.parameters(), lr=1e-3)
        self.dyn_optimizer = torch.optim.Adam(self.dyn_head.parameters(), lr=1e-3)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.total_it = 0
        self.env_step = 0

    def set_env_step(self, step: int):
        self.env_step = int(step)

    def _current_critic_encoder_grad_scale(self) -> float:
        return scheduled_encoder_grad_scale(
            self.critic_encoder_grad_scale,
            self.critic_encoder_grad_schedule,
            self.env_step,
        )

    def select_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        with torch.no_grad():
            latent = self.encoder(state)
            state_input = torch.cat([state, latent * self.latent_input_scale], dim=1)
            action = self.actor(state_input)
        return action.cpu().numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.to(self.device)
        not_done = not_done.to(self.device)

        state_12d = state[:, :12]
        next_state_12d = next_state[:, :12]
        critic_encoder_grad_scale = self._current_critic_encoder_grad_scale()
        critic_updates_encoder = critic_encoder_grad_scale > 0.0

        latent_rep = self.encoder(state)
        recon_loss = F.mse_loss(self.recon_head(latent_rep), state_12d)
        dyn_loss = F.mse_loss(self.dyn_head(latent_rep, action), next_state_12d)
        representation_loss = recon_loss + dyn_loss
        posture_loss, posture_stats = posture_separation_loss(
            state,
            latent_rep,
            margin=self.posture_separation_margin,
        )
        posture_weighted_loss = self.posture_separation_weight * posture_loss
        representation_total_loss = 0.1 * representation_loss + posture_weighted_loss

        with torch.no_grad():
            latent_next_target = self.encoder(next_state)
            next_state_input = torch.cat([next_state, latent_next_target * self.latent_input_scale], dim=1)
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state_input) + noise).clamp(-self.max_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_state_input, next_action)
            target_q = reward + not_done * self.discount * torch.min(target_q1, target_q2)

        if critic_updates_encoder:
            latent_critic = self.encoder(state)
        else:
            with torch.no_grad():
                latent_critic = self.encoder(state)
        latent_critic_rl = soft_detach(latent_critic, critic_encoder_grad_scale)
        state_input = torch.cat([state, latent_critic_rl * self.latent_input_scale], dim=1)

        current_q1, current_q2 = self.critic(state_input, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        total_critic_rep_loss = representation_total_loss + critic_loss

        self.encoder_optimizer.zero_grad()
        self.recon_optimizer.zero_grad()
        self.dyn_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_critic_rep_loss.backward()
        critic_grad_norm = _grad_norm(self.critic.parameters())
        encoder_grad_norm_total = _grad_norm(self.encoder.parameters())
        encoder_grad_norm_rep = encoder_grad_norm_total
        encoder_grad_norm_critic = encoder_grad_norm_total if critic_updates_encoder else 0.0

        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(self.recon_head.parameters(), self.grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(self.dyn_head.parameters(), self.grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)

        self.encoder_optimizer.step()
        self.recon_optimizer.step()
        self.dyn_optimizer.step()
        self.critic_optimizer.step()

        actor_loss_val = None
        actor_grad_norm = 0.0
        encoder_grad_norm_actor = 0.0
        actor_updated = False
        actor_sat_pct = None

        if self.total_it % self.policy_freq == 0:
            if self.actor_encoder_grad_scale > 0.0:
                latent_actor = self.encoder(state)
            else:
                with torch.no_grad():
                    latent_actor = self.encoder(state)
            latent_actor_rl = soft_detach(latent_actor, self.actor_encoder_grad_scale)
            state_input_actor = torch.cat([state, latent_actor_rl * self.latent_input_scale], dim=1)
            actor_loss = -self.critic.Q1(state_input_actor, self.actor(state_input_actor)).mean()

            if self.actor_encoder_grad_scale > 0.0:
                self.encoder_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = _grad_norm(self.actor.parameters())
            if self.actor_encoder_grad_scale > 0.0:
                encoder_grad_norm_actor = _grad_norm(self.encoder.parameters())
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.grad_clip_norm)
                self.encoder_optimizer.step()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
            self.actor_optimizer.step()

            actor_loss_val = float(actor_loss.item())
            actor_updated = True
            with torch.no_grad():
                actor_actions = self.actor(state_input_actor)
                actor_sat_pct = float((actor_actions.abs() >= (self.max_action - 1e-3)).float().mean().item())

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        with torch.no_grad():
            latent_stats = self.encoder(state)

        encoder_grad_norm_total_with_actor = (
            encoder_grad_norm_total**2 + encoder_grad_norm_actor**2
        ) ** 0.5

        return {
            "critic_loss": float(critic_loss.item()),
            "critic_grad_norm": float(critic_grad_norm),
            "recon_loss": float(recon_loss.item()),
            "dyn_loss": float(dyn_loss.item()),
            "representation_loss": float(representation_loss.item()),
            "representation_total_loss": float(representation_total_loss.detach().item()),
            "posture_sep_loss": float(posture_stats["posture_sep_loss"]),
            "posture_sep_weighted_loss": float(posture_weighted_loss.detach().item()),
            "posture_center_distance": float(posture_stats["posture_center_distance"]),
            "posture_hover_count": float(posture_stats["posture_hover_count"]),
            "posture_dive_count": float(posture_stats["posture_dive_count"]),
            "encoder_grad_norm_main": float(encoder_grad_norm_total),
            "encoder_grad_norm_rep": float(encoder_grad_norm_rep),
            "encoder_grad_norm_critic": float(encoder_grad_norm_critic),
            "encoder_grad_norm_critic_recon": float(encoder_grad_norm_rep),
            "encoder_grad_norm_total": float(encoder_grad_norm_total_with_actor),
            "encoder_grad_norm_actor": float(encoder_grad_norm_actor),
            "critic_updates_encoder": float(critic_updates_encoder),
            "actor_updates_encoder": float(self.actor_updates_encoder),
            "critic_encoder_grad_scale": float(critic_encoder_grad_scale),
            "critic_encoder_grad_scale_base": float(self.critic_encoder_grad_scale),
            "actor_encoder_grad_scale": float(self.actor_encoder_grad_scale),
            "env_step": float(self.env_step),
            "latent_input_scale": float(self.latent_input_scale),
            "latent_effective_scale": float(self.latent_input_scale),
            "latent_input_scale_is_zero": float(abs(self.latent_input_scale) <= 1e-12),
            "latent_std_mean": float(latent_stats.std(dim=0).mean().item()),
            "latent_abs_mean": float(latent_stats.abs().mean().item()),
            "latent_input_abs_mean": float((latent_stats * self.latent_input_scale).abs().mean().item()),
            "q1_mean": float(current_q1.detach().mean().item()),
            "q2_mean": float(current_q2.detach().mean().item()),
            "q_target_mean": float(target_q.detach().mean().item()),
            "q1_std": float(current_q1.detach().std(unbiased=False).item()),
            "q2_std": float(current_q2.detach().std(unbiased=False).item()),
            "q_target_std": float(target_q.detach().std(unbiased=False).item()),
            "q_gap_abs_mean": float((current_q1.detach() - current_q2.detach()).abs().mean().item()),
            "actor_loss": actor_loss_val,
            "actor_grad_norm": float(actor_grad_norm),
            "actor_updated": actor_updated,
            "actor_sat_pct": actor_sat_pct,
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
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=self.device))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic)
        self.encoder.load_state_dict(torch.load(filename + "_encoder", map_location=self.device))
        self.encoder_optimizer.load_state_dict(torch.load(filename + "_encoder_optimizer", map_location=self.device))
        self.recon_head.load_state_dict(torch.load(filename + "_recon_head", map_location=self.device))
        self.recon_optimizer.load_state_dict(torch.load(filename + "_recon_optimizer", map_location=self.device))
        self.dyn_head.load_state_dict(torch.load(filename + "_dyn_head", map_location=self.device))
        self.dyn_optimizer.load_state_dict(torch.load(filename + "_dyn_optimizer", map_location=self.device))
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)
