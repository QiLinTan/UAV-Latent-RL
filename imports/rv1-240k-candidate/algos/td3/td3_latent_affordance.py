import copy

import torch
import torch.nn.functional as F

from algos.td3.latent_aux_heads import ActionConditionedAuxHead
from algos.td3.latent_regularizers import KIN_MEAN, KIN_STD, posture_separation_loss
from algos.td3.td3_latent_only import TD3LatentOnly, _grad_norm
from envs.observation_layout import GOAL_DIM, KIN_DIM, RANGE_DIM, ForestObservationLayout


class TD3LatentAffordance(TD3LatentOnly):
    """Latent-only TD3 with weak future-affordance auxiliary targets.

    The auxiliary head predicts action-conditioned task affordances from z:
    long-run goal progress, future clearance, danger probability, and near-goal
    probability. It is intentionally lightweight so the latent bottleneck is
    still primarily shaped by TD3.
    """

    def __init__(
        self,
        *args,
        affordance_loss_weight=0.005,
        affordance_start_step=100000,
        affordance_gamma=0.95,
        affordance_bootstrap_warmup_steps=50000,
        affordance_danger_range=0.20,
        affordance_goal_tolerance=0.20,
        start_pos=(-3.5, 0.0, 1.0),
        **kwargs,
    ):
        state_dim = kwargs.get("state_dim")
        if state_dim is None:
            if not args:
                raise TypeError("TD3LatentAffordance requires state_dim.")
            state_dim = args[0]
        action_dim = kwargs.get("action_dim")
        if action_dim is None:
            if len(args) < 2:
                raise TypeError("TD3LatentAffordance requires action_dim.")
            action_dim = args[1]

        super().__init__(*args, **kwargs)
        self.affordance_loss_weight = float(affordance_loss_weight)
        self.affordance_start_step = int(affordance_start_step)
        self.affordance_gamma = float(affordance_gamma)
        self.affordance_bootstrap_warmup_steps = int(affordance_bootstrap_warmup_steps)
        self.affordance_danger_range = float(affordance_danger_range)
        self.affordance_goal_tolerance = float(affordance_goal_tolerance)
        self.start_pos = tuple(float(x) for x in start_pos)
        self.observation_layout = ForestObservationLayout.from_total_dim(
            int(state_dim),
            action_dim=int(action_dim),
        )

        self.affordance_head = ActionConditionedAuxHead(
            latent_dim=self.latent_dim,
            action_dim=int(action_dim),
            output_dim=4,
        ).to(self.device)
        self.affordance_head_target = copy.deepcopy(self.affordance_head)
        self.affordance_optimizer = torch.optim.Adam(self.affordance_head.parameters(), lr=1e-3)

    def _position(self, state: torch.Tensor) -> torch.Tensor:
        kin_mean = state.new_tensor(KIN_MEAN[:3])
        kin_std = state.new_tensor(KIN_STD[:3])
        return state[:, :3] * kin_std + kin_mean

    def _route_len(self, state: torch.Tensor) -> torch.Tensor:
        start_pos = state.new_tensor(self.start_pos).view(1, 3)
        target_pos = state.new_tensor(self.target_pos).view(1, 3)
        return torch.norm(target_pos[:, :2] - start_pos[:, :2], dim=1, keepdim=True).clamp_min(1e-6)

    def _range_values(self, state: torch.Tensor) -> torch.Tensor:
        return self.observation_layout.split(state).ranges

    def _range_min(self, state: torch.Tensor) -> torch.Tensor:
        return self._range_values(state).clamp(0.0, 1.0).min(dim=1, keepdim=True).values

    @staticmethod
    def _std(value: torch.Tensor) -> float:
        return float(value.detach().std(unbiased=False).item())

    @staticmethod
    def _abs_mean_or_zero(value: torch.Tensor) -> float:
        if value.numel() == 0:
            return 0.0
        return float(value.detach().abs().mean().item())

    def _observation_diagnostics(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
    ) -> dict[str, float]:
        current = self.observation_layout.split(state)
        following = self.observation_layout.split(next_state)

        def range_stats(values: torch.Tensor, prefix: str):
            per_sample_min = values.min(dim=1).values
            per_sample_max = values.max(dim=1).values
            out_of_bounds = ((values < 0.0) | (values > 1.0)).float()
            return {
                f"{prefix}_range_min_mean": float(per_sample_min.mean().item()),
                f"{prefix}_range_max_mean": float(per_sample_max.mean().item()),
                f"{prefix}_range_oob_fraction": float(out_of_bounds.mean().item()),
            }

        diagnostics = {
            "obs_kin_abs_mean": self._abs_mean_or_zero(current.kin),
            "obs_action_history_abs_mean": self._abs_mean_or_zero(current.action_history),
            "obs_goal_norm_mean": float(torch.linalg.vector_norm(current.goal, dim=1).mean().item()),
            "next_obs_goal_norm_mean": float(
                torch.linalg.vector_norm(following.goal, dim=1).mean().item()
            ),
        }
        diagnostics.update(range_stats(current.ranges, "obs"))
        diagnostics.update(range_stats(following.ranges, "next_obs"))
        return diagnostics

    def _affordance_scale(self) -> float:
        return 1.0 if self.env_step >= self.affordance_start_step else 0.0

    def _affordance_bootstrap_scale(self) -> float:
        if self.env_step < self.affordance_start_step:
            return 0.0
        if self.affordance_bootstrap_warmup_steps <= 0:
            return 1.0
        elapsed = self.env_step - self.affordance_start_step
        return min(1.0, max(0.0, elapsed / float(self.affordance_bootstrap_warmup_steps)))

    def _weighted_clearance_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per_sample = F.smooth_l1_loss(pred, target, reduction="none")
        weights = 1.0 + 2.0 * (target < 0.35).float() + 4.0 * (target < self.affordance_danger_range).float()
        return (per_sample * weights).mean()

    def _affordance_predictions(self, z: torch.Tensor, action: torch.Tensor):
        raw = self.affordance_head(z, action)
        return {
            "raw": raw,
            "progress_value": raw[:, 0:1],
            "future_clearance": torch.sigmoid(raw[:, 1:2]),
            "danger_logit": raw[:, 2:3],
            "near_goal_logit": raw[:, 3:4],
        }

    def _affordance_targets(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        not_done: torch.Tensor,
        delta_goal: torch.Tensor,
    ):
        route_len = self._route_len(state)
        progress_step = (delta_goal / route_len).clamp(-0.25, 0.25)
        next_goal_distance = self._goal_distance(next_state)
        immediate_success = (next_goal_distance < self.affordance_goal_tolerance).float()
        range_min_next = self._range_min(next_state)

        with torch.no_grad():
            latent_next = self.encoder(next_state)
            next_action = self.actor_target(self._policy_input(latent_next))
            next_raw = self.affordance_head_target(latent_next, next_action)
            bootstrap_scale = self._affordance_bootstrap_scale()

            next_progress_value = next_raw[:, 0:1].clamp(-1.0, 1.0)
            progress_value = (
                progress_step
                + bootstrap_scale * self.affordance_gamma * not_done * next_progress_value
            ).clamp(-1.0, 1.0)

            next_near_goal_prob = torch.sigmoid(next_raw[:, 3:4])
            near_goal_prob = torch.maximum(
                immediate_success,
                bootstrap_scale * self.affordance_gamma * not_done * next_near_goal_prob,
            ).clamp(0.0, 1.0)

            next_future_clearance = torch.sigmoid(next_raw[:, 1:2])
            bootstrapped_clearance = torch.where(
                not_done > 0.5,
                torch.minimum(range_min_next, next_future_clearance),
                range_min_next,
            )
            future_clearance = (
                bootstrapped_clearance if bootstrap_scale > 0.0 else range_min_next
            ).clamp(0.0, 1.0)

            immediate_danger = (range_min_next < self.affordance_danger_range).float()
            next_danger_prob = torch.sigmoid(next_raw[:, 2:3])
            danger_prob = torch.maximum(
                immediate_danger,
                bootstrap_scale * self.affordance_gamma * not_done * next_danger_prob,
            ).clamp(0.0, 1.0)

        return {
            "progress_value": progress_value,
            "future_clearance": future_clearance,
            "danger_prob": danger_prob,
            "near_goal_prob": near_goal_prob,
            "progress_step": progress_step,
            "range_min_next": range_min_next,
            "immediate_danger": immediate_danger,
            "immediate_success": immediate_success,
        }

    def _soft_update_affordance_target(self):
        for param, target_param in zip(self.affordance_head.parameters(), self.affordance_head_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

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
        progress_pred = self.progress_head(latent_rep, action)
        with torch.no_grad():
            goal_distance = self._goal_distance(state)
            next_goal_distance = self._goal_distance(next_state)
            delta_goal = goal_distance - next_goal_distance
        progress_loss = F.mse_loss(progress_pred, delta_goal)

        affordance_pred = self._affordance_predictions(latent_rep, action)
        affordance_target = self._affordance_targets(state, next_state, not_done, delta_goal)
        affordance_progress_value_loss = F.smooth_l1_loss(
            affordance_pred["progress_value"],
            affordance_target["progress_value"],
        )
        affordance_clearance_loss = self._weighted_clearance_loss(
            affordance_pred["future_clearance"],
            affordance_target["future_clearance"],
        )
        affordance_danger_loss = F.binary_cross_entropy_with_logits(
            affordance_pred["danger_logit"],
            affordance_target["danger_prob"],
        )
        affordance_near_goal_loss = F.binary_cross_entropy_with_logits(
            affordance_pred["near_goal_logit"],
            affordance_target["near_goal_prob"],
        )
        affordance_loss = (
            affordance_progress_value_loss
            + affordance_clearance_loss
            + 0.5 * affordance_danger_loss
            + 0.5 * affordance_near_goal_loss
        )

        representation_loss = recon_loss + dyn_loss
        posture_loss, posture_stats = posture_separation_loss(
            state,
            latent_rep,
            margin=self.posture_separation_margin,
        )
        posture_weighted_loss = self.posture_separation_weight * posture_loss
        progress_weighted_loss = self.progress_loss_weight * progress_loss
        affordance_scale = self._affordance_scale()
        affordance_weighted_loss = affordance_scale * self.affordance_loss_weight * affordance_loss
        representation_total_loss = (
            0.1 * representation_loss
            + progress_weighted_loss
            + posture_weighted_loss
            + affordance_weighted_loss
        )

        with torch.no_grad():
            latent_next_target = self.encoder(next_state)
            next_latent_input = self._policy_input(latent_next_target)
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_latent_input) + noise).clamp(-self.max_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_latent_input, next_action)
            target_q = reward + not_done * self.discount * torch.min(target_q1, target_q2)

        if critic_updates_encoder:
            latent_critic = self.encoder(state)
        else:
            with torch.no_grad():
                latent_critic = self.encoder(state)
        latent_input = self._policy_input(
            latent_critic,
            latent_grad_scale=critic_encoder_grad_scale,
        )
        current_q1, current_q2 = self.critic(latent_input, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        total_critic_rep_loss = representation_total_loss + critic_loss

        self.encoder_optimizer.zero_grad()
        self.recon_optimizer.zero_grad()
        self.dyn_optimizer.zero_grad()
        self.progress_optimizer.zero_grad()
        self.affordance_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_critic_rep_loss.backward()
        critic_grad_norm = _grad_norm(self.critic.parameters())
        encoder_grad_norm_total = _grad_norm(self.encoder.parameters())
        encoder_grad_norm_rep = encoder_grad_norm_total
        encoder_grad_norm_critic = encoder_grad_norm_total if critic_updates_encoder else 0.0

        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(self.recon_head.parameters(), self.grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(self.dyn_head.parameters(), self.grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(self.progress_head.parameters(), self.grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(self.affordance_head.parameters(), self.grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.encoder_optimizer.step()
        self.recon_optimizer.step()
        self.dyn_optimizer.step()
        self.progress_optimizer.step()
        self.affordance_optimizer.step()
        self.critic_optimizer.step()
        self._soft_update_affordance_target()

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
            latent_input_actor = self._policy_input(
                latent_actor,
                latent_grad_scale=self.actor_encoder_grad_scale,
            )
            actor_loss = -self.critic.Q1(latent_input_actor, self.actor(latent_input_actor)).mean()

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
                actor_actions = self.actor(latent_input_actor)
                actor_sat_pct = float((actor_actions.abs() >= (self.max_action - 1e-3)).float().mean().item())

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        with torch.no_grad():
            latent_stats = self.encoder(state)
            danger_prob_pred = torch.sigmoid(affordance_pred["danger_logit"])
            near_goal_prob_pred = torch.sigmoid(affordance_pred["near_goal_logit"])
            observation_diagnostics = self._observation_diagnostics(state, next_state)

        encoder_grad_norm_total_with_actor = (
            encoder_grad_norm_total**2 + encoder_grad_norm_actor**2
        ) ** 0.5

        future_clearance_target = affordance_target["future_clearance"]
        danger_prob_target = affordance_target["danger_prob"]
        range_min_next = affordance_target["range_min_next"]
        train_info = {
            "critic_loss": float(critic_loss.item()),
            "critic_grad_norm": float(critic_grad_norm),
            "recon_loss": float(recon_loss.item()),
            "dyn_loss": float(dyn_loss.item()),
            "progress_loss": float(progress_loss.item()),
            "progress_weighted_loss": float(progress_weighted_loss.detach().item()),
            "progress_pred_mean": float(progress_pred.detach().mean().item()),
            "progress_target_mean": float(delta_goal.mean().item()),
            "goal_distance_mean": float(goal_distance.mean().item()),
            "affordance_loss": float(affordance_loss.detach().item()),
            "affordance_weighted_loss": float(affordance_weighted_loss.detach().item()),
            "affordance_progress_value_loss": float(affordance_progress_value_loss.detach().item()),
            "affordance_clearance_loss": float(affordance_clearance_loss.detach().item()),
            "affordance_danger_loss": float(affordance_danger_loss.detach().item()),
            "affordance_near_goal_loss": float(affordance_near_goal_loss.detach().item()),
            "affordance_progress_value_pred_mean": float(affordance_pred["progress_value"].detach().mean().item()),
            "affordance_progress_value_target_mean": float(affordance_target["progress_value"].detach().mean().item()),
            "affordance_progress_step_target_mean": float(affordance_target["progress_step"].detach().mean().item()),
            "affordance_future_clearance_pred_mean": float(affordance_pred["future_clearance"].detach().mean().item()),
            "affordance_progress_value_target_std": self._std(affordance_target["progress_value"]),
            "affordance_future_clearance_target_mean": float(
                future_clearance_target.detach().mean().item()
            ),
            "affordance_future_clearance_target_std": self._std(future_clearance_target),
            "affordance_danger_prob_pred_mean": float(danger_prob_pred.detach().mean().item()),
            "affordance_danger_prob_target_mean": float(danger_prob_target.detach().mean().item()),
            "affordance_danger_prob_target_std": self._std(danger_prob_target),
            "affordance_near_goal_prob_pred_mean": float(near_goal_prob_pred.detach().mean().item()),
            "affordance_near_goal_prob_target_mean": float(affordance_target["near_goal_prob"].detach().mean().item()),
            "affordance_near_goal_prob_target_std": self._std(affordance_target["near_goal_prob"]),
            "affordance_next_min_range_mean": float(range_min_next.detach().mean().item()),
            "affordance_immediate_danger_rate": float(
                affordance_target["immediate_danger"].detach().mean().item()
            ),
            "affordance_immediate_success_rate": float(
                affordance_target["immediate_success"].detach().mean().item()
            ),
            "affordance_scale": float(affordance_scale),
            "affordance_bootstrap_scale": float(self._affordance_bootstrap_scale()),
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
            "affordance_loss_weight": float(self.affordance_loss_weight),
            "affordance_start_step": float(self.affordance_start_step),
            "affordance_gamma": float(self.affordance_gamma),
            "affordance_bootstrap_warmup_steps": float(self.affordance_bootstrap_warmup_steps),
            "affordance_danger_range": float(self.affordance_danger_range),
            "affordance_goal_tolerance": float(self.affordance_goal_tolerance),
            "obs_total_dim": float(self.observation_layout.total_dim),
            "obs_kin_dim": float(KIN_DIM),
            "obs_action_history_dim": float(self.observation_layout.action_history_dim),
            "obs_goal_dim": float(GOAL_DIM),
            "obs_range_dim": float(RANGE_DIM),
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
        train_info.update(observation_diagnostics)
        return train_info

    def save(self, filename):
        super().save(filename)
        torch.save(self.affordance_head.state_dict(), filename + "_affordance_head")
        torch.save(self.affordance_head_target.state_dict(), filename + "_affordance_head_target")
        torch.save(self.affordance_optimizer.state_dict(), filename + "_affordance_optimizer")

    def load(self, filename):
        super().load(filename)
        try:
            self.affordance_head.load_state_dict(torch.load(filename + "_affordance_head", map_location=self.device))
            self.affordance_optimizer.load_state_dict(
                torch.load(filename + "_affordance_optimizer", map_location=self.device)
            )
            try:
                self.affordance_head_target.load_state_dict(
                    torch.load(filename + "_affordance_head_target", map_location=self.device)
                )
            except FileNotFoundError:
                self.affordance_head_target = copy.deepcopy(self.affordance_head)
        except FileNotFoundError:
            self.affordance_head_target = copy.deepcopy(self.affordance_head)
