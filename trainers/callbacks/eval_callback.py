import numpy as np
import contextlib
import json
import os
import sys
import time

from envs.preprocess import preprocess_state
from gym_pybullet_drones.utils.utils import sync
from trainers.navigation_metrics import episode_navigation_metrics
from trainers.upper_trajectory_export import export_upper_trajectory


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


def _trajectory_obstacle_specs(env):
    """Return visualization metadata for either legacy trees or atlas geometry."""

    map_spec = getattr(env, "MAP_SPEC", None)
    if map_spec is not None and map_spec.obstacles:
        specs = []
        for obstacle in map_spec.obstacles:
            spec = {
                "kind": obstacle.kind,
                "name": obstacle.name,
                "position": np.asarray(obstacle.position, dtype=np.float32).copy(),
            }
            if obstacle.radius is not None:
                spec["radius"] = float(obstacle.radius)
            if obstacle.height is not None:
                spec["height"] = float(obstacle.height)
            if obstacle.half_extents is not None:
                spec["half_extents"] = np.asarray(
                    obstacle.half_extents, dtype=np.float32
                ).copy()
            specs.append(spec)
        return specs

    return [
        {
            "kind": "cylinder",
            "name": "tree",
            "position": np.array(
                [spec["xy"][0], spec["xy"][1], float(spec["height"]) / 2.0],
                dtype=np.float32,
            ),
            "radius": float(spec["radius"]),
            "height": float(spec["height"]),
        }
        for spec in getattr(env, "_tree_specs", ())
    ]


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
        output_dir=None,
        trajectory_output_dir=None,
    ):
        self.env_cls = env_cls
        self.env_kwargs = dict(env_kwargs)
        self.interval = interval
        self.episodes = episodes
        self.step_sleep = step_sleep
        self.eval_gui = bool(eval_gui)
        self.quiet = bool(quiet)
        self.output_dir = None if output_dir is None else os.fspath(output_dir)
        self.trajectory_output_dir = (
            None if trajectory_output_dir is None else os.fspath(trajectory_output_dir)
        )
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    def on_step(self, trainer, *, force=False):
        if not force and trainer.total_steps % self.interval != 0:
            return

        curriculum_metrics = {}
        if hasattr(trainer.env, "get_curriculum_metrics"):
            curriculum_metrics = dict(trainer.env.get_curriculum_metrics())
        curriculum_metrics["replay_size"] = int(getattr(trainer.buffer, "size", 0))

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
        navigation_metrics = []
        episode_reward_terms = []
        seed_zero_trace = None
        seed_zero_obstacle_specs = None
        seed_zero_start = None
        seed_zero_target = None
        seed_zero_episode_summary = None

        if not self.quiet:
            print(f"[Eval @ {trainer.total_steps}] Starting evaluation...")

        with _suppress_native_output(self.quiet):
            eval_env = self.env_cls(**self.env_kwargs, gui=self.eval_gui)
            try:
                if (
                    bool(getattr(eval_env, "CURRICULUM_ENABLED", False))
                    and hasattr(eval_env, "set_curriculum_stage_override")
                    and hasattr(trainer.env, "get_curriculum_stage")
                ):
                    eval_env.set_curriculum_stage_override(trainer.env.get_curriculum_stage())
                for ep in range(self.episodes):
                    obs, reset_info = eval_env.reset(seed=ep)
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
                    ep_reward_terms = {}
                    ep_positions = [
                        np.asarray(eval_env.pos[0], dtype=np.float32).copy()
                    ]
                    ep_goal_distances = [
                        float((reset_info or {}).get("goal_distance", np.nan))
                    ]
                    capture_trajectory = self.trajectory_output_dir is not None and ep == 0
                    if capture_trajectory:
                        initial_position = np.asarray(eval_env.pos[0], dtype=np.float32).copy()
                        initial_velocity = np.asarray(eval_env.vel[0], dtype=np.float32).copy()
                        initial_reference_position = np.asarray(
                            getattr(eval_env, "reference_position", initial_position),
                            dtype=np.float32,
                        ).copy()
                        initial_reference_velocity = np.asarray(
                            getattr(eval_env, "reference_velocity", initial_velocity),
                            dtype=np.float32,
                        ).copy()
                        seed_zero_trace = {
                            "time_seconds": [0.0],
                            "actual_position": [initial_position],
                            "actual_velocity": [initial_velocity],
                            "reference_position": [initial_reference_position],
                            "reference_velocity": [initial_reference_velocity],
                            "packet_endpoint": [initial_reference_position.copy()],
                            "upper_action": [],
                            "reward": [],
                            "goal_distance": [],
                            "min_tree_clearance": [],
                            "packet_version": [],
                            "packet_valid": [],
                        }
                        seed_zero_obstacle_specs = _trajectory_obstacle_specs(eval_env)
                        seed_zero_start = np.asarray(
                            getattr(eval_env, "START_POS", initial_position), dtype=np.float32
                        ).copy()
                        seed_zero_target = np.asarray(
                            getattr(eval_env, "TARGET_POS", initial_position), dtype=np.float32
                        ).copy()

                    while not done:
                        action = trainer.agent.select_action(state)
                        obs, reward, terminated, truncated, info = eval_env.step(action.reshape(1, -1))
                        state = preprocess_state(obs.reshape(-1))
                        done = terminated or truncated
                        ep_ret += float(reward)
                        step_idx += 1
                        last_info = info or {}
                        last_truncated = bool(truncated)
                        for key, value in last_info.items():
                            if key.startswith("reward/") and np.isscalar(value):
                                term_name = key[len("reward/") :]
                                ep_reward_terms[term_name] = (
                                    ep_reward_terms.get(term_name, 0.0) + float(value)
                                )
                        ep_positions.append(
                            np.asarray(eval_env.pos[0], dtype=np.float32).copy()
                        )
                        ep_goal_distances.append(
                            float(last_info.get("goal_distance", np.nan))
                        )

                        if capture_trajectory:
                            packet = getattr(eval_env, "last_reference_packet", None)
                            reference_position = np.asarray(
                                getattr(eval_env, "reference_position", eval_env.pos[0]),
                                dtype=np.float32,
                            ).copy()
                            reference_velocity = np.asarray(
                                getattr(eval_env, "reference_velocity", eval_env.vel[0]),
                                dtype=np.float32,
                            ).copy()
                            packet_endpoint = reference_position.copy()
                            if packet is not None:
                                packet_endpoint = np.asarray(
                                    packet.to_world(packet.positions[-1]), dtype=np.float32
                                ).copy()
                            seed_zero_trace["time_seconds"].append(
                                float(step_idx) * float(eval_env.CTRL_TIMESTEP)
                            )
                            seed_zero_trace["actual_position"].append(
                                np.asarray(eval_env.pos[0], dtype=np.float32).copy()
                            )
                            seed_zero_trace["actual_velocity"].append(
                                np.asarray(eval_env.vel[0], dtype=np.float32).copy()
                            )
                            seed_zero_trace["reference_position"].append(reference_position)
                            seed_zero_trace["reference_velocity"].append(reference_velocity)
                            seed_zero_trace["packet_endpoint"].append(packet_endpoint)
                            seed_zero_trace["upper_action"].append(
                                np.asarray(action, dtype=np.float32).copy()
                            )
                            seed_zero_trace["reward"].append(float(reward))
                            seed_zero_trace["goal_distance"].append(
                                float(last_info.get("goal_distance", np.nan))
                            )
                            seed_zero_trace["min_tree_clearance"].append(
                                float(last_info.get("min_tree_clearance", np.nan))
                            )
                            seed_zero_trace["packet_version"].append(
                                int(last_info.get("reference_packet_version", -1))
                            )
                            seed_zero_trace["packet_valid"].append(
                                bool(last_info.get("reference_packet_valid", False))
                            )

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
                    metrics = episode_navigation_metrics(
                        ep_positions,
                        ep_goal_distances,
                        timestep_seconds=float(eval_env.CTRL_TIMESTEP),
                    )
                    navigation_metrics.append(metrics)
                    episode_reward_terms.append(ep_reward_terms)
                    if capture_trajectory:
                        seed_zero_episode_summary = {
                            "return": float(ep_ret),
                            "length": int(step_idx),
                            "final_goal_distance": float(last_info.get("goal_distance", np.nan)),
                            "success": bool(last_info.get("success", False)),
                            "collision": bool(last_info.get("collision", False)),
                            "truncated": bool(last_truncated),
                            "done_reason": str(last_info.get("done_reason", "unknown")),
                            "max_roll_pitch": float(ep_max_attitude),
                            "reward_terms": dict(ep_reward_terms),
                            **metrics,
                        }
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
                **{
                    f"eval/{key}": float(np.mean([item[key] for item in navigation_metrics]))
                    for key in (
                        "path_length",
                        "path_efficiency",
                        "backtrack_ratio",
                        "max_no_progress_seconds",
                        "stuck_window_fraction",
                        "revisit_rate",
                    )
                },
            }
            trainer.last_eval_info.update(
                {f"eval/{key}": value for key, value in curriculum_metrics.items()}
            )
            reward_term_names = sorted(
                {
                    key
                    for episode_terms in episode_reward_terms
                    for key in episode_terms
                }
            )
            trainer.last_eval_info.update(
                {
                    f"eval/reward/{key}": float(
                        np.mean([terms.get(key, 0.0) for terms in episode_reward_terms])
                    )
                    for key in reward_term_names
                }
            )
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
            if self.output_dir is not None:
                record = {
                    "step": int(trainer.total_steps),
                    "curriculum": curriculum_metrics,
                    "summary": {
                        key: None if value is None else float(value)
                        for key, value in trainer.last_eval_info.items()
                    },
                    "episodes": [
                        {
                            "episode": int(index),
                            "return": float(returns[index]),
                            "length": int(lengths[index]),
                            "final_goal_distance": float(final_dists[index]),
                            "success": bool(successes[index]),
                            "collision": bool(collisions[index]),
                            "truncated": bool(truncated_flags[index]),
                            "done_reason": done_reasons[index],
                            "max_roll_pitch": float(max_attitudes[index]),
                            "reward_terms": episode_reward_terms[index],
                            **navigation_metrics[index],
                        }
                        for index in range(len(lengths))
                    ],
                }
                with open(
                    os.path.join(self.output_dir, "evaluation_metrics.jsonl"),
                    "a",
                    encoding="utf-8",
                ) as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (
                self.trajectory_output_dir is not None
                and seed_zero_trace is not None
                and seed_zero_episode_summary is not None
            ):
                exported = export_upper_trajectory(
                    self.trajectory_output_dir,
                    seed=0,
                    training_step=trainer.total_steps,
                    trace=seed_zero_trace,
                    tree_specs=seed_zero_obstacle_specs,
                    start_position=seed_zero_start,
                    target_position=seed_zero_target,
                    episode_summary=seed_zero_episode_summary,
                    evaluation_summary=trainer.last_eval_info,
                )
                print(
                    f"[Trajectory @ {trainer.total_steps}] "
                    f"xy={exported['xy']} | xyz={exported['xyz']}"
                )

    def on_episode_end(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass
