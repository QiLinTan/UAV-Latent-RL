import copy

import torch
import torch.nn.functional as F

from algos.td3.networks import Actor, Critic
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


def _per_sample_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean(dim=1)


def compute_samplewise_trust(
    rec_err_per_sample: torch.Tensor,
    dyn_err_per_sample: torch.Tensor,
    rec_err_ema: float,
    dyn_err_ema: float,
    alpha: float = 0.5,
    beta: float = 0.5,
    q_min: float = 0.05,
    q_max: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    rec_rel = rec_err_per_sample / (rec_err_ema + eps)
    dyn_rel = dyn_err_per_sample / (dyn_err_ema + eps)
    trust = torch.exp(-(alpha * rec_rel + beta * dyn_rel))
    return trust.clamp(q_min, q_max)


def compute_recon_trust(
    rec_err_per_sample: torch.Tensor,
    rec_err_ema: float,
    q_min: float = 0.05,
    q_max: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    rec_rel = rec_err_per_sample / (rec_err_ema + eps)
    trust = torch.exp(-rec_rel)
    return trust.clamp(q_min, q_max)


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
        latent_input_scale=0.1,
        grad_clip_norm=1.0,
        latent_dim=16,
        trust_alpha=0.5,
        trust_beta=0.5,
        trust_q_min=0.05,
        trust_q_max=1.0,
        trust_ema_momentum=0.99,
        trust_eps=1e-6,
        trust_warmup_steps=10000,
        device=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_action = float(max_action)

        self.latent_dim = int(latent_dim)
        self.latent_input_scale = float(latent_input_scale)
        self.grad_clip_norm = float(grad_clip_norm)
        self.actor_updates_encoder = bool(actor_updates_encoder)
        self.discount = float(discount)
        self.tau = float(tau)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.policy_freq = int(policy_freq)
        self.trust_alpha = float(trust_alpha)
        self.trust_beta = float(trust_beta)
        self.trust_q_min = float(trust_q_min)
        self.trust_q_max = float(trust_q_max)
        self.trust_ema_momentum = float(trust_ema_momentum)
        self.trust_eps = float(trust_eps)
        self.trust_warmup_steps = max(0, int(trust_warmup_steps))
        self.rec_err_ema = 1.0
        self.dyn_err_ema = 1.0

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

    def _warmup_mix(self, trust: torch.Tensor) -> torch.Tensor:
        if self.trust_warmup_steps <= 0:
            return trust
        warmup_scale = min(1.0, float(self.total_it) / float(self.trust_warmup_steps))
        return self.trust_q_min + warmup_scale * (trust - self.trust_q_min)

    def _build_state_input(
        self,
        state: torch.Tensor,
        latent: torch.Tensor,
        trust: torch.Tensor,
    ) -> torch.Tensor:
        trust = trust.detach().unsqueeze(1)
        latent_weighted = latent.detach() * (self.latent_input_scale * trust)
        return torch.cat([state, latent_weighted], dim=1)

    def _compute_action_trust(self, state: torch.Tensor):
        latent = self.encoder(state)
        state_12d = state[:, :12]
        state_recon = self.recon_head(latent)
        rec_err_per_sample = _per_sample_mse(state_recon, state_12d)
        trust = compute_recon_trust(
            rec_err_per_sample=rec_err_per_sample,
            rec_err_ema=self.rec_err_ema,
            q_min=self.trust_q_min,
            q_max=self.trust_q_max,
            eps=self.trust_eps,
        )
        trust = self._warmup_mix(trust).detach()
        return latent.detach(), trust, rec_err_per_sample.detach()

    def select_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        with torch.no_grad():
            latent, trust, _ = self._compute_action_trust(state)
            state_input = self._build_state_input(state, latent, trust)
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

        # Step A: update representation only from reconstruction / dynamics losses.
        latent = self.encoder(state)
        state_recon = self.recon_head(latent)
        next_state_pred = self.dyn_head(latent, action)
        rec_err_per_sample = _per_sample_mse(state_recon, state_12d)
        dyn_err_per_sample = _per_sample_mse(next_state_pred, next_state_12d)
        recon_loss = rec_err_per_sample.mean()
        dyn_loss = dyn_err_per_sample.mean()
        representation_loss = recon_loss + dyn_loss

        self.encoder_optimizer.zero_grad()
        self.recon_optimizer.zero_grad()
        self.dyn_optimizer.zero_grad()
        representation_loss.backward()
        encoder_grad_norm_rep = _grad_norm(self.encoder.parameters())

        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(self.recon_head.parameters(), self.grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(self.dyn_head.parameters(), self.grad_clip_norm)

        self.encoder_optimizer.step()
        self.recon_optimizer.step()
        self.dyn_optimizer.step()

        # Step B: recompute detached latent / trust with updated representation.
        with torch.no_grad():
            latent_detached = self.encoder(state)
            latent_next_detached = self.encoder(next_state)

            state_recon_detached = self.recon_head(latent_detached)
            next_state_recon_detached = self.recon_head(latent_next_detached)
            next_state_pred_detached = self.dyn_head(latent_detached, action)

            rec_err_per_sample = _per_sample_mse(state_recon_detached, state_12d)
            dyn_err_per_sample = _per_sample_mse(next_state_pred_detached, next_state_12d)
            next_rec_err_per_sample = _per_sample_mse(next_state_recon_detached, next_state_12d)

            batch_rec_mean = float(rec_err_per_sample.mean().item())
            batch_dyn_mean = float(dyn_err_per_sample.mean().item())
            self.rec_err_ema = (
                self.trust_ema_momentum * self.rec_err_ema
                + (1.0 - self.trust_ema_momentum) * batch_rec_mean
            )
            self.dyn_err_ema = (
                self.trust_ema_momentum * self.dyn_err_ema
                + (1.0 - self.trust_ema_momentum) * batch_dyn_mean
            )

            trust = compute_samplewise_trust(
                rec_err_per_sample=rec_err_per_sample,
                dyn_err_per_sample=dyn_err_per_sample,
                rec_err_ema=self.rec_err_ema,
                dyn_err_ema=self.dyn_err_ema,
                alpha=self.trust_alpha,
                beta=self.trust_beta,
                q_min=self.trust_q_min,
                q_max=self.trust_q_max,
                eps=self.trust_eps,
            )
            trust_next = compute_recon_trust(
                rec_err_per_sample=next_rec_err_per_sample,
                rec_err_ema=self.rec_err_ema,
                q_min=self.trust_q_min,
                q_max=self.trust_q_max,
                eps=self.trust_eps,
            )
            trust = self._warmup_mix(trust).detach()
            trust_next = self._warmup_mix(trust_next).detach()

            state_input = self._build_state_input(state, latent_detached, trust)
            next_state_input = self._build_state_input(next_state, latent_next_detached, trust_next)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state_input) + noise).clamp(-self.max_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_state_input, next_action)
            target_q = reward + not_done * self.discount * torch.min(target_q1, target_q2)

        current_q1, current_q2 = self.critic(state_input, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)

        self.critic_optimizer.step()

        actor_loss_val = None
        encoder_grad_norm_actor = 0.0
        actor_updated = False
        actor_sat_pct = None

        if self.total_it % self.policy_freq == 0:
            state_input_actor = self._build_state_input(state, latent_detached, trust)
            actor_loss = -self.critic.Q1(state_input_actor, self.actor(state_input_actor)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
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

        return {
            "critic_loss": float(critic_loss.item()),
            "recon_loss": float(recon_loss.item()),
            "dyn_loss": float(dyn_loss.item()),
            "representation_loss": float(representation_loss.item()),
            "encoder_grad_norm_rep": float(encoder_grad_norm_rep),
            "encoder_grad_norm_critic_recon": float(encoder_grad_norm_rep),
            "encoder_grad_norm_actor": float(encoder_grad_norm_actor),
            "latent_std_mean": float(latent_detached.std(dim=0).mean().item()),
            "trust_mean": float(trust.mean().item()),
            "trust_min": float(trust.min().item()),
            "trust_max": float(trust.max().item()),
            "trust_next_mean": float(trust_next.mean().item()),
            "rec_err_ema": float(self.rec_err_ema),
            "dyn_err_ema": float(self.dyn_err_ema),
            "actor_loss": actor_loss_val,
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
        torch.save(
            {
                "rec_err_ema": float(self.rec_err_ema),
                "dyn_err_ema": float(self.dyn_err_ema),
                "total_it": int(self.total_it),
            },
            filename + "_trust_stats",
        )

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
        try:
            trust_stats = torch.load(filename + "_trust_stats", map_location=self.device)
            self.rec_err_ema = float(trust_stats.get("rec_err_ema", self.rec_err_ema))
            self.dyn_err_ema = float(trust_stats.get("dyn_err_ema", self.dyn_err_ema))
            self.total_it = int(trust_stats.get("total_it", self.total_it))
        except FileNotFoundError:
            pass
