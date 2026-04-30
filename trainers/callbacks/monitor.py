import numpy as np

def _physics_monitor_from_env(env, last_action: np.ndarray):
    """
    Physics monitor based on BaseAviary internal state.
    Note: last_action is the [-1, 1] normalized RPM action output by the actor/exploration (one dimension per motor).
    """
    target = np.asarray(getattr(env, "TARGET_POS", [0.0, 0.0, 1.0]), dtype=np.float32)
    pos = env.pos[0].astype(np.float32)
    rpy = env.rpy[0].astype(np.float32)
    vel = env.vel[0].astype(np.float32)

    last_action = np.asarray(last_action, dtype=np.float32).reshape(-1)
    motor_rpms = (env.HOVER_RPM * (1.0 + 0.05 * last_action)).astype(np.float32)
    motor_rpms = np.clip(motor_rpms, 0.0, env.MAX_RPM).astype(np.float32)
    sat_pct = (motor_rpms / env.MAX_RPM * 100.0).astype(np.float32)

    front_mean = float(motor_rpms[[0, 1]].mean())
    back_mean = float(motor_rpms[[2, 3]].mean())
    motor_front_back_diff = abs(front_mean - back_mean)

    action_sat = float((np.abs(last_action) >= (1.0 - 1e-3)).mean() * 100.0)

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

        monitor = physics_monitor(env, last_action)
        train_info = getattr(trainer, "last_train_info", {}) or {}
        last_info = getattr(trainer, "last_info", {}) or {}
        critic_loss = train_info.get("critic_loss", None)
        dyn_loss = train_info.get("dyn_loss", None)
        actor_sat_pct = train_info.get("actor_sat_pct", None)
        trust_mean = train_info.get("trust_mean", None)
        rec_err_ema = train_info.get("rec_err_ema", None)
        dyn_err_ema = train_info.get("dyn_err_ema", None)
        train_steps_this_tick = getattr(trainer, "train_steps_this_tick", 0)
        step = trainer.total_steps

        rpms_list = [f"{x:.0f}" for x in monitor["motor_rpms"]]
        sat_list = [f"{x:.0f}" for x in monitor["sat_pct"]]
        r_deg, p_deg, y_deg = monitor["rpy_deg"]

        critic_loss_str = "N/A" if critic_loss is None else f"{critic_loss:.3f}"
        dyn_loss_str = "N/A" if dyn_loss is None else f"{dyn_loss:.3f}"
        actor_sat_pct_str = "N/A" if actor_sat_pct is None else f"{actor_sat_pct:.3f}"
        trust_mean_str = "N/A" if trust_mean is None else f"{trust_mean:.3f}"
        rec_err_ema_str = "N/A" if rec_err_ema is None else f"{rec_err_ema:.4f}"
        dyn_err_ema_str = "N/A" if dyn_err_ema is None else f"{dyn_err_ema:.4f}"

        print(f"====================== Physics Monitor @ STEP {step} ======================")
        print(
            f"Critic loss: {critic_loss_str} | dyn_loss: {dyn_loss_str} | "
            f"trust_mean: {trust_mean_str}"
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
            f"Action saturation: {monitor['action_sat_pct']:.1f}% | train_steps_this_tick: {train_steps_this_tick} | "
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
