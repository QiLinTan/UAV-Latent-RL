import numpy as np

def _physics_monitor_from_env(env, last_action: np.ndarray, action_limit: float = 1.0):
    """
    Physics monitor based on BaseAviary internal state.
    Note: last_action is the normalized RPM action output by the actor/exploration (one dimension per motor).
    """
    target = np.asarray(getattr(env, "TARGET_POS", [0.0, 0.0, 1.0]), dtype=np.float32)
    pos = env.pos[0].astype(np.float32)
    rpy = env.rpy[0].astype(np.float32)
    vel = env.vel[0].astype(np.float32)

    last_action = np.asarray(last_action, dtype=np.float32).reshape(-1)
    codec = getattr(env, "motor_codec", None)
    if codec is None:
        codec = getattr(env, "motor_action_codec", None)
    if codec is not None:
        motor_rpms = np.asarray(
            codec.normalized_action_to_rpm(last_action),
            dtype=np.float32,
        )
    else:
        motor_rpms = (env.HOVER_RPM * (1.0 + 0.05 * last_action)).astype(np.float32)
    motor_rpms = np.clip(motor_rpms, 0.0, env.MAX_RPM).astype(np.float32)
    sat_pct = (motor_rpms / env.MAX_RPM * 100.0).astype(np.float32)

    front_mean = float(motor_rpms[[0, 1]].mean())
    back_mean = float(motor_rpms[[2, 3]].mean())
    motor_front_back_diff = abs(front_mean - back_mean)

    action_limit = max(float(action_limit), 1e-6)
    action_sat = float((np.abs(last_action) >= (action_limit - 1e-3)).mean() * 100.0)

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
        "action_limit": action_limit,
        "avg_thrust": avg_thrust,
    }

physics_monitor = _physics_monitor_from_env

class MonitorCallback:
    def __init__(self, interval=10000):
        self.interval = interval

    def on_step(self, trainer):
        if trainer.total_steps % self.interval != 0:
            return

        env = trainer.env
        last_action = getattr(trainer, "last_action", None)

        if last_action is None:
            return

        if np.asarray(last_action).size != 4 and hasattr(env, "last_motor_action"):
            last_action = env.last_motor_action

        monitor = physics_monitor(env, last_action, action_limit=getattr(trainer, "max_action", 1.0))
        train_info = getattr(trainer, "last_train_info", {}) or {}
        last_info = getattr(trainer, "last_info", {}) or {}
        critic_loss = train_info.get("critic_loss", None)
        dyn_loss = train_info.get("dyn_loss", None)
        recon_loss = train_info.get("recon_loss", None)
        progress_loss = train_info.get("progress_loss", None)
        progress_target_mean = train_info.get("progress_target_mean", None)
        route_action_loss = train_info.get("route_action_loss", None)
        safety_action_loss = train_info.get("safety_action_loss", None)
        action_aux_scale = train_info.get("action_aux_scale", None)
        route_next_lateral_target_mean = train_info.get("route_next_lateral_target_mean", None)
        safety_next_min_range_target_mean = train_info.get("safety_next_min_range_target_mean", None)
        affordance_loss = train_info.get("affordance_loss", None)
        affordance_scale = train_info.get("affordance_scale", None)
        affordance_bootstrap_scale = train_info.get("affordance_bootstrap_scale", None)
        affordance_progress_target_mean = train_info.get("affordance_progress_value_target_mean", None)
        affordance_clearance_target_mean = train_info.get("affordance_future_clearance_target_mean", None)
        affordance_danger_pred_mean = train_info.get("affordance_danger_prob_pred_mean", None)
        affordance_danger_target_mean = train_info.get("affordance_danger_prob_target_mean", None)
        affordance_near_goal_target_mean = train_info.get("affordance_near_goal_prob_target_mean", None)
        affordance_immediate_danger_rate = train_info.get("affordance_immediate_danger_rate", None)
        affordance_immediate_success_rate = train_info.get("affordance_immediate_success_rate", None)
        obs_action_history_dim = train_info.get("obs_action_history_dim", None)
        obs_goal_norm_mean = train_info.get("obs_goal_norm_mean", None)
        obs_range_min_mean = train_info.get("obs_range_min_mean", None)
        obs_range_max_mean = train_info.get("obs_range_max_mean", None)
        obs_range_oob_fraction = train_info.get("obs_range_oob_fraction", None)
        next_obs_range_min_mean = train_info.get("next_obs_range_min_mean", None)
        critic_grad_norm = train_info.get("critic_grad_norm", None)
        actor_grad_norm = train_info.get("actor_grad_norm", None)
        encoder_grad_norm_main = train_info.get("encoder_grad_norm_main", train_info.get("encoder_grad_norm_total", None))
        encoder_grad_norm_rep = train_info.get("encoder_grad_norm_rep", None)
        encoder_grad_norm_critic = train_info.get("encoder_grad_norm_critic", None)
        encoder_grad_norm_actor = train_info.get("encoder_grad_norm_actor", None)
        actor_sat_pct = train_info.get("actor_sat_pct", None)
        trust_mean = train_info.get("trust_mean", None)
        rec_err_ema = train_info.get("rec_err_ema", None)
        dyn_err_ema = train_info.get("dyn_err_ema", None)
        latent_input_scale = train_info.get("latent_input_scale", None)
        latent_effective_scale = train_info.get("latent_effective_scale", None)
        latent_input_abs_mean = train_info.get("latent_input_abs_mean", None)
        actor_updates_encoder = train_info.get("actor_updates_encoder", None)
        critic_updates_encoder = train_info.get("critic_updates_encoder", None)
        actor_encoder_grad_scale = train_info.get("actor_encoder_grad_scale", None)
        critic_encoder_grad_scale = train_info.get("critic_encoder_grad_scale", None)
        train_steps_this_tick = getattr(trainer, "train_steps_this_tick", 0)
        step = trainer.total_steps

        rpms_list = [f"{x:.0f}" for x in monitor["motor_rpms"]]
        sat_list = [f"{x:.0f}" for x in monitor["sat_pct"]]
        r_deg, p_deg, y_deg = monitor["rpy_deg"]

        critic_loss_str = "N/A" if critic_loss is None else f"{critic_loss:.3f}"
        dyn_loss_str = "N/A" if dyn_loss is None else f"{dyn_loss:.3f}"
        recon_loss_str = "N/A" if recon_loss is None else f"{recon_loss:.3f}"
        progress_loss_str = "N/A" if progress_loss is None else f"{progress_loss:.5f}"
        progress_target_mean_str = "N/A" if progress_target_mean is None else f"{progress_target_mean:+.5f}"
        route_action_loss_str = "N/A" if route_action_loss is None else f"{route_action_loss:.5f}"
        safety_action_loss_str = "N/A" if safety_action_loss is None else f"{safety_action_loss:.5f}"
        action_aux_scale_str = "N/A" if action_aux_scale is None else f"{action_aux_scale:.1f}"
        route_next_lateral_target_mean_str = (
            "N/A" if route_next_lateral_target_mean is None else f"{route_next_lateral_target_mean:.4f}"
        )
        safety_next_min_range_target_mean_str = (
            "N/A" if safety_next_min_range_target_mean is None else f"{safety_next_min_range_target_mean:.4f}"
        )
        affordance_loss_str = "N/A" if affordance_loss is None else f"{affordance_loss:.5f}"
        affordance_scale_str = "N/A" if affordance_scale is None else f"{affordance_scale:.1f}"
        affordance_bootstrap_scale_str = (
            "N/A" if affordance_bootstrap_scale is None else f"{affordance_bootstrap_scale:.2f}"
        )
        affordance_progress_target_mean_str = (
            "N/A" if affordance_progress_target_mean is None else f"{affordance_progress_target_mean:+.4f}"
        )
        affordance_clearance_target_mean_str = (
            "N/A" if affordance_clearance_target_mean is None else f"{affordance_clearance_target_mean:.4f}"
        )
        affordance_danger_pred_mean_str = (
            "N/A" if affordance_danger_pred_mean is None else f"{affordance_danger_pred_mean:.4f}"
        )
        affordance_danger_target_mean_str = (
            "N/A" if affordance_danger_target_mean is None else f"{affordance_danger_target_mean:.4f}"
        )
        affordance_near_goal_target_mean_str = (
            "N/A" if affordance_near_goal_target_mean is None else f"{affordance_near_goal_target_mean:.4f}"
        )
        affordance_immediate_danger_rate_str = (
            "N/A" if affordance_immediate_danger_rate is None else f"{affordance_immediate_danger_rate:.3f}"
        )
        affordance_immediate_success_rate_str = (
            "N/A" if affordance_immediate_success_rate is None else f"{affordance_immediate_success_rate:.3f}"
        )
        obs_action_history_dim_str = (
            "N/A" if obs_action_history_dim is None else f"{int(obs_action_history_dim)}"
        )
        obs_goal_norm_mean_str = (
            "N/A" if obs_goal_norm_mean is None else f"{obs_goal_norm_mean:.3f}"
        )
        obs_range_min_mean_str = (
            "N/A" if obs_range_min_mean is None else f"{obs_range_min_mean:.3f}"
        )
        obs_range_max_mean_str = (
            "N/A" if obs_range_max_mean is None else f"{obs_range_max_mean:.3f}"
        )
        obs_range_oob_fraction_str = (
            "N/A" if obs_range_oob_fraction is None else f"{obs_range_oob_fraction:.3f}"
        )
        next_obs_range_min_mean_str = (
            "N/A" if next_obs_range_min_mean is None else f"{next_obs_range_min_mean:.3f}"
        )
        critic_grad_norm_str = "N/A" if critic_grad_norm is None else f"{critic_grad_norm:.3f}"
        actor_grad_norm_str = "N/A" if actor_grad_norm is None else f"{actor_grad_norm:.3f}"
        encoder_grad_norm_main_str = "N/A" if encoder_grad_norm_main is None else f"{encoder_grad_norm_main:.3f}"
        encoder_grad_norm_rep_str = "N/A" if encoder_grad_norm_rep is None else f"{encoder_grad_norm_rep:.3f}"
        encoder_grad_norm_critic_str = "N/A" if encoder_grad_norm_critic is None else f"{encoder_grad_norm_critic:.3f}"
        encoder_grad_norm_actor_str = "N/A" if encoder_grad_norm_actor is None else f"{encoder_grad_norm_actor:.3f}"
        actor_sat_pct_str = "N/A" if actor_sat_pct is None else f"{actor_sat_pct:.3f}"
        trust_mean_str = "N/A" if trust_mean is None else f"{trust_mean:.3f}"
        rec_err_ema_str = "N/A" if rec_err_ema is None else f"{rec_err_ema:.4f}"
        dyn_err_ema_str = "N/A" if dyn_err_ema is None else f"{dyn_err_ema:.4f}"
        latent_input_scale_str = "N/A" if latent_input_scale is None else f"{latent_input_scale:.4f}"
        latent_effective_scale_str = "N/A" if latent_effective_scale is None else f"{latent_effective_scale:.4f}"
        latent_input_abs_mean_str = "N/A" if latent_input_abs_mean is None else f"{latent_input_abs_mean:.4f}"
        actor_updates_encoder_str = "N/A" if actor_updates_encoder is None else f"{bool(actor_updates_encoder)}"
        critic_updates_encoder_str = "N/A" if critic_updates_encoder is None else f"{bool(critic_updates_encoder)}"
        actor_encoder_grad_scale_str = "N/A" if actor_encoder_grad_scale is None else f"{actor_encoder_grad_scale:.3f}"
        critic_encoder_grad_scale_str = "N/A" if critic_encoder_grad_scale is None else f"{critic_encoder_grad_scale:.3f}"

        print(f"====================== Physics Monitor @ STEP {step} ======================")
        print(
            f"Critic loss: {critic_loss_str} | recon_loss: {recon_loss_str} | dyn_loss: {dyn_loss_str} | "
            f"trust_mean: {trust_mean_str}"
        )
        print(f"Progress head: loss={progress_loss_str} | target_delta_goal_mean={progress_target_mean_str}")
        if route_action_loss is not None or safety_action_loss is not None or action_aux_scale is not None:
            print(
                f"Action-aux heads: scale={action_aux_scale_str} | route_loss={route_action_loss_str} | "
                f"safety_loss={safety_action_loss_str} | next_lateral={route_next_lateral_target_mean_str} | "
                f"next_min_range={safety_next_min_range_target_mean_str}"
            )
        if affordance_loss is not None:
            print(
                f"Affordance head: scale={affordance_scale_str} | bootstrap={affordance_bootstrap_scale_str} | "
                f"loss={affordance_loss_str} | progress_v={affordance_progress_target_mean_str} | "
                f"future_clearance={affordance_clearance_target_mean_str} | "
                f"danger={affordance_danger_pred_mean_str}/{affordance_danger_target_mean_str} | "
                f"near_goal={affordance_near_goal_target_mean_str} | "
                f"immediate danger/success={affordance_immediate_danger_rate_str}/"
                f"{affordance_immediate_success_rate_str}"
            )
            print(
                f"Observation layout: action_hist={obs_action_history_dim_str} | "
                f"goal_norm={obs_goal_norm_mean_str} | "
                f"range min/max={obs_range_min_mean_str}/{obs_range_max_mean_str} | "
                f"next_min={next_obs_range_min_mean_str} | range_oob={obs_range_oob_fraction_str}"
            )
        print(
            f"Grad norms: critic={critic_grad_norm_str} | actor={actor_grad_norm_str} | "
            f"encoder(main/actor)={encoder_grad_norm_main_str}/{encoder_grad_norm_actor_str}"
        )
        print(
            f"Latent input: scale={latent_input_scale_str} | effective={latent_effective_scale_str} | "
            f"abs_mean={latent_input_abs_mean_str}"
        )
        print(
            f"Encoder RL grad scale: critic={critic_encoder_grad_scale_str} | actor={actor_encoder_grad_scale_str} | "
            f"critic_updates_encoder={critic_updates_encoder_str} | actor_updates_encoder={actor_updates_encoder_str}"
        )
        print(f"Position: x={monitor['pos'][0]:+.2f}, y={monitor['pos'][1]:+.2f}, z={monitor['pos'][2]:+.2f}")
        print(f"Attitude(deg): R={r_deg:+.1f}, P={p_deg:+.1f}, Y={y_deg:+.1f}")
        print(f"Z velocity: {monitor['z_vel']:+.3f} | Target distance: {monitor['dist']:+.2f}")
        if last_info:
            curriculum_stage = last_info.get("curriculum_stage", "N/A")
            corridor_half_width = last_info.get("corridor_half_width", None)
            reward_progress = last_info.get("reward/progress_reward", None)
            height_penalty = last_info.get("reward/height_penalty", None)
            lateral_penalty = last_info.get("reward/lateral_penalty", None)
            proximity_penalty = last_info.get("reward/proximity_penalty", None)
            clearance = last_info.get("min_tree_clearance", None)
            corridor_str = "N/A" if corridor_half_width is None else f"{float(corridor_half_width):.2f}"
            reward_progress_str = "N/A" if reward_progress is None else f"{float(reward_progress):+.3f}"
            height_penalty_str = "N/A" if height_penalty is None else f"{float(height_penalty):.3f}"
            lateral_penalty_str = "N/A" if lateral_penalty is None else f"{float(lateral_penalty):.3f}"
            proximity_penalty_str = "N/A" if proximity_penalty is None else f"{float(proximity_penalty):.3f}"
            clearance_str = "N/A" if clearance is None else f"{float(clearance):.3f}"
            print(
                f"Curriculum stage: {curriculum_stage} | corridor_half_width: {corridor_str} | "
                f"clearance: {clearance_str}"
            )
            print(
                f"Reward terms: progress={reward_progress_str} | height_penalty={height_penalty_str} | "
                f"lateral_penalty={lateral_penalty_str} | proximity_penalty={proximity_penalty_str}"
            )
        print(f"Motor RPMs: [{', '.join(rpms_list)}] | sat%: [{', '.join(sat_list)}]%")
        print(f"Front-back motor speed difference: {monitor['motor_front_back_diff']:.0f} RPM")
        print(
            f"Action saturation(@{monitor['action_limit']:.2f}): {monitor['action_sat_pct']:.1f}% | "
            f"train_steps_this_tick: {train_steps_this_tick} | "
            f"avg_thrust: {monitor['avg_thrust']:.3f} | actor_sat_pct(train): {actor_sat_pct_str}"
        )
        print(
            f"Trust EMA stats: rec_err_ema={rec_err_ema_str} | dyn_err_ema={dyn_err_ema_str}"
        )
        print("========================================================================")

    def on_episode_end(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass
