from numbers import Real

from torch.utils.tensorboard import SummaryWriter


class LoggerCallback:
    CORE_TRAIN_KEYS = frozenset(
        {
            "critic_loss",
            "actor_loss",
            "recon_loss",
            "dyn_loss",
            "progress_loss",
            "route_action_loss",
            "safety_action_loss",
            "action_aux_weighted_loss",
            "affordance_loss",
            "affordance_weighted_loss",
            "affordance_progress_value_loss",
            "affordance_clearance_loss",
            "affordance_danger_loss",
            "affordance_near_goal_loss",
            "q1_mean",
            "q2_mean",
            "q_target_mean",
            "q_gap_abs_mean",
            "trust_mean",
            "trust_next_mean",
            "rec_err_ema",
            "dyn_err_ema",
        }
    )
    LATENT_TRAIN_KEYS = frozenset(
        {
            "latent_std_mean",
            "latent_abs_mean",
            "latent_input_abs_mean",
            "latent_effective_scale",
            "posture_center_distance",
            "posture_sep_loss",
            "route_delta_progress_target_mean",
            "route_next_lateral_target_mean",
            "route_next_lateral_pred_mean",
            "route_next_xy_distance_target_mean",
            "route_next_xy_distance_pred_mean",
            "safety_next_min_range_target_mean",
            "safety_next_min_range_pred_mean",
            "action_aux_scale",
            "affordance_progress_value_target_mean",
            "affordance_progress_value_target_std",
            "affordance_progress_value_pred_mean",
            "affordance_progress_step_target_mean",
            "affordance_future_clearance_target_mean",
            "affordance_future_clearance_target_std",
            "affordance_future_clearance_pred_mean",
            "affordance_danger_prob_target_mean",
            "affordance_danger_prob_target_std",
            "affordance_danger_prob_pred_mean",
            "affordance_near_goal_prob_target_mean",
            "affordance_near_goal_prob_target_std",
            "affordance_near_goal_prob_pred_mean",
            "affordance_next_min_range_mean",
            "affordance_immediate_danger_rate",
            "affordance_immediate_success_rate",
            "affordance_scale",
            "affordance_bootstrap_scale",
            "obs_kin_abs_mean",
            "obs_action_history_abs_mean",
            "obs_goal_norm_mean",
            "obs_range_min_mean",
            "obs_range_max_mean",
            "obs_range_oob_fraction",
            "next_obs_goal_norm_mean",
            "next_obs_range_min_mean",
            "next_obs_range_max_mean",
            "next_obs_range_oob_fraction",
        }
    )
    CONFIG_KEYS = frozenset(
        {
            "actor_updates_encoder",
            "critic_updates_encoder",
            "critic_encoder_grad_scale_base",
            "actor_encoder_grad_scale",
            "latent_input_scale",
            "latent_input_scale_is_zero",
            "route_action_loss_weight",
            "safety_action_loss_weight",
            "action_aux_start_step",
            "affordance_loss_weight",
            "affordance_start_step",
            "affordance_gamma",
            "affordance_bootstrap_warmup_steps",
            "affordance_danger_range",
            "affordance_goal_tolerance",
            "obs_total_dim",
            "obs_kin_dim",
            "obs_action_history_dim",
            "obs_goal_dim",
            "obs_range_dim",
        }
    )
    SKIP_SCALAR_KEYS = frozenset({"env_step"})

    def __init__(self, log_dir, train_interval=500, debug_interval=1000):
        self.writer = SummaryWriter(log_dir)
        self.train_interval = self._normalize_interval(train_interval)
        self.debug_interval = self._normalize_interval(debug_interval)
        self._latest_train_info = {}
        self._last_logged_eval_step = None
        self._logged_train_config = False
        self.writer.add_text(
            "config/logging",
            "\n".join(
                [
                    "| setting | value |",
                    "| --- | --- |",
                    f"| train_interval | {self.train_interval or 'disabled'} |",
                    f"| debug_interval | {self.debug_interval or 'disabled'} |",
                ]
            ),
            0,
        )

    @staticmethod
    def _normalize_interval(interval):
        if interval is None:
            return None
        interval = int(interval)
        return interval if interval > 0 else None

    @staticmethod
    def _to_scalar(value):
        if value is None:
            return None
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, Real):
            return float(value)
        return None

    @staticmethod
    def _should_log(step, interval):
        return interval is not None and step % interval == 0

    def _write_scalars(self, values, keys, step):
        for key in sorted(keys):
            scalar = values.get(key)
            if scalar is not None:
                self.writer.add_scalar(key, scalar, step)

    def _log_train_config_once(self, step):
        config_values = {
            key: self._latest_train_info[key]
            for key in sorted(self.CONFIG_KEYS)
            if key in self._latest_train_info
        }
        if not config_values:
            self._logged_train_config = True
            return
        rows = ["| key | value |", "| --- | --- |"]
        rows.extend(f"| `{key}` | {value:g} |" for key, value in config_values.items())
        self.writer.add_text("config/train_info", "\n".join(rows), step)
        self._logged_train_config = True

    def on_step(self, trainer):
        """记录训练信息（损失值等）"""
        train_info = getattr(trainer, "last_train_info", None)
        if train_info:
            for key, value in train_info.items():
                scalar = self._to_scalar(value)
                if scalar is not None:
                    self._latest_train_info[key] = scalar

            if not self._logged_train_config:
                self._log_train_config_once(trainer.total_steps)

            core_keys = self.CORE_TRAIN_KEYS | self.LATENT_TRAIN_KEYS
            if self._should_log(trainer.total_steps, self.train_interval):
                self._write_scalars(self._latest_train_info, core_keys, trainer.total_steps)

            if self._should_log(trainer.total_steps, self.debug_interval):
                debug_keys = (
                    set(self._latest_train_info)
                    - core_keys
                    - self.CONFIG_KEYS
                    - self.SKIP_SCALAR_KEYS
                )
                self._write_scalars(self._latest_train_info, debug_keys, trainer.total_steps)

        eval_info = getattr(trainer, "last_eval_info", None)
        eval_step = getattr(trainer, "last_eval_step", None)
        if eval_info and eval_step is not None and eval_step != self._last_logged_eval_step:
            for k, v in eval_info.items():
                scalar = self._to_scalar(v)
                if scalar is not None:
                    self.writer.add_scalar(k, scalar, eval_step)
            self._last_logged_eval_step = eval_step

    def on_episode_end(self, trainer):
        """记录 episode 级别的指标"""
        # 记录返回值
        self.writer.add_scalar("episode/return", trainer.episode_return, trainer.total_steps)
        self.writer.add_scalar("episode/length", trainer.episode_step, trainer.total_steps)
        self.writer.add_scalar("episode/avg_reward", 
                              trainer.episode_return / max(1, trainer.episode_step), 
                              trainer.total_steps)
        
        # 记录环境信息（如果有）
        if hasattr(trainer, "last_info") and trainer.last_info:
            for k, v in trainer.last_info.items():
                val = self._to_scalar(v)
                if val is not None:
                    self.writer.add_scalar(f"env/{k}", val, trainer.total_steps)

    def on_train_end(self, trainer):
        eval_info = getattr(trainer, "last_eval_info", None)
        eval_step = getattr(trainer, "last_eval_step", None)
        if eval_info and eval_step is not None and eval_step != self._last_logged_eval_step:
            for k, v in eval_info.items():
                scalar = self._to_scalar(v)
                if scalar is not None:
                    self.writer.add_scalar(k, scalar, eval_step)
            self._last_logged_eval_step = eval_step
        self.writer.close()
