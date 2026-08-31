import numpy as np
import contextlib
import os
import sys
import time

from envs.preprocess import preprocess_state
from gym_pybullet_drones.utils.utils import sync


@contextlib.contextmanager
def _suppress_native_output(enabled: bool):
    if not enabled:
        yield
        return

    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


class EvalCallback:
    def __init__(
        self,
        env_cls,
        env_kwargs,
        interval=10000,
        episodes=1,
        step_sleep=True,
        eval_gui=True,
        quiet=True,
    ):
        self.env_cls = env_cls
        self.env_kwargs = dict(env_kwargs)
        self.interval = interval
        self.episodes = episodes
        self.step_sleep = step_sleep
        self.eval_gui = bool(eval_gui)
        self.quiet = bool(quiet)

    def on_step(self, trainer):
        if trainer.total_steps % self.interval != 0:
            return

        returns = []
        lengths = []
        min_zs = []
        final_zs = []
        mean_abs_z_errors = []
        mean_z_vels = []
        final_dists = []
        successes = []
        collisions = []
        truncated_flags = []
        done_reasons = []
        tracking_position_errors = []
        tracking_velocity_errors = []
        max_attitudes = []

        if not self.quiet:
            print(f"[Eval @ {trainer.total_steps}] Starting evaluation...")

        with _suppress_native_output(self.quiet):
            eval_env = self.env_cls(**self.env_kwargs, gui=self.eval_gui)
            try:
                if hasattr(eval_env, "set_curriculum_stage_override") and hasattr(trainer.env, "get_curriculum_stage"):
                    eval_env.set_curriculum_stage_override(trainer.env.get_curriculum_stage())
                for ep in range(self.episodes):
                    obs, _ = eval_env.reset(seed=ep)
                    state = preprocess_state(obs.reshape(-1))
                    if hasattr(trainer.agent, "reset_episode"):
                        trainer.agent.reset_episode()
                    if hasattr(trainer.agent, "configure_motor_action_interface"):
                        trainer.agent.configure_motor_action_interface(eval_env)
                    done = False
                    ep_ret = 0.0
                    start_time = time.time()
                    step_idx = 0
                    ep_zs = []
                    ep_z_vels = []
                    ep_z_errors = []
                    last_info = {}
                    last_truncated = False
                    ep_tracking_position_errors = []
                    ep_tracking_velocity_errors = []
                    ep_max_attitude = 0.0

                    while not done:
                        action = trainer.agent.select_action(state)
                        obs, reward, terminated, truncated, info = eval_env.step(action.reshape(1, -1))
                        state = preprocess_state(obs.reshape(-1))
                        done = terminated or truncated
                        ep_ret += float(reward)
                        step_idx += 1
                        last_info = info or {}
                        last_truncated = bool(truncated)

                        if hasattr(eval_env, "pos") and hasattr(eval_env, "vel"):
                            z = float(eval_env.pos[0][2])
                            z_vel = float(eval_env.vel[0][2])
                            target_z = float(getattr(eval_env, "TARGET_POS", [0.0, 0.0, 1.0])[2])
                            ep_zs.append(z)
                            ep_z_vels.append(z_vel)
                            ep_z_errors.append(abs(z - target_z))
                            reference_sample = getattr(
                                trainer.agent,
                                "last_reference_sample",
                                None,
                            )
                            if reference_sample is None:
                                reference_sample = getattr(
                                    eval_env,
                                    "last_reference_sample",
                                    None,
                                )
                            if reference_sample is not None:
                                ep_tracking_position_errors.append(
                                    float(
                                        np.linalg.norm(
                                            eval_env.pos[0]
                                            - reference_sample["lookahead_position"]
                                        )
                                    )
                                )
                                ep_tracking_velocity_errors.append(
                                    float(
                                        np.linalg.norm(
                                            eval_env.vel[0]
                                            - reference_sample["lookahead_velocity"]
                                        )
                                    )
                                )
                            if hasattr(eval_env, "rpy"):
                                ep_max_attitude = max(
                                    ep_max_attitude,
                                    abs(float(eval_env.rpy[0][0])),
                                    abs(float(eval_env.rpy[0][1])),
                                )

                        if self.step_sleep:
                            sync(step_idx, start_time, eval_env.CTRL_TIMESTEP)

                    returns.append(ep_ret)
                    lengths.append(step_idx)
                    if ep_zs:
                        min_zs.append(float(np.min(ep_zs)))
                        final_zs.append(float(ep_zs[-1]))
                        mean_abs_z_errors.append(float(np.mean(ep_z_errors)))
                        mean_z_vels.append(float(np.mean(ep_z_vels)))
                    final_dists.append(float(last_info.get("goal_distance", np.nan)))
                    successes.append(float(last_info.get("success", False)))
                    collisions.append(float(last_info.get("collision", False)))
                    truncated_flags.append(float(last_truncated))
                    done_reasons.append(str(last_info.get("done_reason", "unknown")))
                    if ep_tracking_position_errors:
                        tracking_position_errors.append(
                            float(np.mean(ep_tracking_position_errors))
                        )
                        tracking_velocity_errors.append(
                            float(np.mean(ep_tracking_velocity_errors))
                        )
                    max_attitudes.append(ep_max_attitude)
            finally:
                eval_env.close()

        eval_return = float(np.mean(returns))
        trainer.last_eval_return = eval_return
        print(f"[Eval @ {trainer.total_steps}] return={eval_return:.2f}")
        if lengths:
            trainer.last_eval_info = {
                "eval/return": eval_return,
                "eval/length": float(np.mean(lengths)),
                "eval/min_z": float(np.mean(min_zs)) if min_zs else float("nan"),
                "eval/final_z": float(np.mean(final_zs)) if final_zs else float("nan"),
                "eval/mean_abs_z_error": float(np.mean(mean_abs_z_errors)) if mean_abs_z_errors else float("nan"),
                "eval/mean_z_vel": float(np.mean(mean_z_vels)) if mean_z_vels else float("nan"),
                "eval/final_goal_distance": float(np.nanmean(final_dists)) if final_dists else float("nan"),
                "eval/success_rate": float(np.mean(successes)),
                "eval/collision_rate": float(np.mean(collisions)),
                "eval/truncated_rate": float(np.mean(truncated_flags)),
                "eval/done_reason_success_rate": float(
                    np.mean([reason == "success" for reason in done_reasons])
                ),
                "eval/done_reason_collision_rate": float(
                    np.mean([reason == "collision" for reason in done_reasons])
                ),
                "eval/done_reason_attitude_bound_rate": float(
                    np.mean([reason == "attitude_bound" for reason in done_reasons])
                ),
                "eval/done_reason_height_bound_rate": float(
                    np.mean([reason == "height_bound" for reason in done_reasons])
                ),
                "eval/done_reason_xy_bound_rate": float(
                    np.mean([reason == "xy_bound" for reason in done_reasons])
                ),
                "eval/done_reason_timeout_rate": float(
                    np.mean([reason == "timeout" for reason in done_reasons])
                ),
                "eval/tracking_position_rmse": (
                    float(np.mean(tracking_position_errors))
                    if tracking_position_errors
                    else float("nan")
                ),
                "eval/tracking_velocity_rmse": (
                    float(np.mean(tracking_velocity_errors))
                    if tracking_velocity_errors
                    else float("nan")
                ),
                "eval/max_roll_pitch": (
                    float(np.mean(max_attitudes)) if max_attitudes else float("nan")
                ),
            }
            trainer.last_eval_step = trainer.total_steps
            print(
                f"[Eval @ {trainer.total_steps}] "
                f"len={trainer.last_eval_info['eval/length']:.1f} | "
                f"min_z={trainer.last_eval_info['eval/min_z']:.2f} | "
                f"final_z={trainer.last_eval_info['eval/final_z']:.2f} | "
                f"mean_z_err={trainer.last_eval_info['eval/mean_abs_z_error']:.2f} | "
                f"mean_z_vel={trainer.last_eval_info['eval/mean_z_vel']:+.3f} | "
                f"final_dist={trainer.last_eval_info['eval/final_goal_distance']:.2f} | "
                f"track_rmse={trainer.last_eval_info['eval/tracking_position_rmse']:.2f} | "
                f"max_att={trainer.last_eval_info['eval/max_roll_pitch']:.2f} | "
                f"success={trainer.last_eval_info['eval/success_rate']:.2f} | "
                f"truncated={trainer.last_eval_info['eval/truncated_rate']:.2f}"
            )

    def on_episode_end(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass
