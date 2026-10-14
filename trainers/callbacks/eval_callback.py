import numpy as np
import contextlib
import json
import os
import sys
import time
from collections import deque

from envs.preprocess import preprocess_state
from gym_pybullet_drones.utils.utils import sync
from trainers.collision_diagnostics import WINDOW_SECONDS, classify_collision_window
from trainers.navigation_metrics import episode_navigation_metrics
from trainers.upper_eval_diagnostics import (
    Z_ACTION_SCAN_VALUES,
    p_height_ablation_action,
    z_action_candidates,
)
from trainers.timeout_diagnostics import classify_timeout
from trainers.terminal_action_diagnostics import (
    export_action_interface_audit,
    export_xy_q_scan,
    xy_action_candidates,
)
from trainers.terminal_coverage_diagnostics import (
    export_terminal_coverage,
    summarize_terminal_coverage,
    summarize_terminal_episode,
)
from trainers.upper_trajectory_export import export_upper_trajectory
from trainers.paired_trajectory_trace import PairedTrajectoryTrace


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


class Stage01PairedEvalCallback:
    """Evaluate retention and adaptation on separate, unmixed map sets."""

    def __init__(self, stage0_callback, stage1_callback):
        self.stage0_callback = stage0_callback
        self.stage1_callback = stage1_callback

    def on_step(self, trainer, *, force=False):
        if not force and trainer.total_steps % self.stage1_callback.interval != 0:
            return
        self.stage0_callback.on_step(trainer, force=True)
        retention = dict(trainer.last_eval_info or {})
        self.stage1_callback.on_step(trainer, force=True)
        trainer.last_eval_info.update({
            "eval/stage0_success_rate": retention["eval/success_rate"],
            "eval/stage0_collision_rate": retention["eval/collision_rate"],
        })

    def on_episode_end(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass


class FullFTTripleEvalCallback:
    """Primary Full Stage 3 validation plus Full Stage 0 and K1 retention."""

    def __init__(self, full_stage3, full_stage0, k1_stage3):
        self.full_stage3 = full_stage3
        self.full_stage0 = full_stage0
        self.k1_stage3 = k1_stage3
        self.interval = full_stage3.interval

    def on_step(self, trainer, *, force=False):
        if not force and trainer.total_steps % self.interval != 0:
            return
        self.full_stage3.on_step(trainer, force=True)
        primary = dict(trainer.last_eval_info or {})
        self.full_stage0.on_step(trainer, force=True)
        stage0 = dict(trainer.last_eval_info or {})
        self.k1_stage3.on_step(trainer, force=True)
        k1 = dict(trainer.last_eval_info or {})
        trainer.last_eval_info = {
            **primary,
            "eval/full_stage3_success_rate": primary["eval/success_rate"],
            "eval/full_stage3_collision_rate": primary["eval/collision_rate"],
            "eval/stage0_success_rate": stage0["eval/success_rate"],
            "eval/full_stage0_collision_rate": stage0["eval/collision_rate"],
            "eval/k1_stage3_success_rate": k1["eval/success_rate"],
            "eval/k1_stage3_collision_rate": k1["eval/collision_rate"],
        }
        trainer.last_eval_step = trainer.total_steps

    def on_episode_end(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass


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
        collision_diagnostic_output_dir=None,
        timeout_diagnostic_output_dir=None,
        p_height_ablation=False,
        p_height_gain=1.0,
        z_q_scan_output_dir=None,
        action_interface_audit_output_dir=None,
        terminal_xy_q_scan_output_dir=None,
        terminal_coverage_output_dir=None,
        fixed_curriculum_stage=None,
        episode_seed_offset=0,
        episode_seeds=None,
        paired_trace_output_dir=None,
        paired_trace_step_dir=False,
        fixed_command_tape_dir=None,
        fixed_command_steps=None,
        command_intervention_start_step=None,
        command_intervention_steps=None,
        eval_episode_len_sec=None,
    ):
        self.env_cls = env_cls
        self.env_kwargs = dict(env_kwargs)
        self.fixed_curriculum_stage = fixed_curriculum_stage
        self.episode_seed_offset = int(episode_seed_offset)
        self.episode_seeds = None if episode_seeds is None else tuple(int(seed) for seed in episode_seeds)
        if self.episode_seeds is not None and not self.episode_seeds:
            raise ValueError("episode_seeds must be nonempty")
        self.paired_trace_output_dir = None if paired_trace_output_dir is None else os.fspath(paired_trace_output_dir)
        self.paired_trace_step_dir = bool(paired_trace_step_dir)
        self.fixed_command_tape_dir = None if fixed_command_tape_dir is None else os.fspath(fixed_command_tape_dir)
        self.fixed_command_steps = None if fixed_command_steps is None else int(fixed_command_steps)
        if self.fixed_command_steps is not None and self.fixed_command_steps <= 0:
            raise ValueError("fixed_command_steps must be positive")
        self.command_intervention_start_step = (
            None if command_intervention_start_step is None else int(command_intervention_start_step)
        )
        self.command_intervention_steps = (
            None if command_intervention_steps is None else int(command_intervention_steps)
        )
        if self.command_intervention_start_step is not None:
            if self.fixed_command_tape_dir is None or self.command_intervention_start_step < 0:
                raise ValueError("Command intervention requires a tape and nonnegative start step")
            if self.command_intervention_steps is None or self.command_intervention_steps <= 0:
                raise ValueError("Command intervention requires positive step count")
            if self.fixed_command_steps is not None:
                raise ValueError("Full-tape limit and local command intervention are mutually exclusive")
        elif self.command_intervention_steps is not None:
            raise ValueError("Command intervention step count requires a start step")
        self.eval_episode_len_sec = None if eval_episode_len_sec is None else float(eval_episode_len_sec)
        if self.eval_episode_len_sec is not None and self.eval_episode_len_sec <= 0:
            raise ValueError("Evaluation episode length must be positive")
        self.interval = interval
        self.episodes = episodes
        self.step_sleep = step_sleep
        self.eval_gui = bool(eval_gui)
        self.quiet = bool(quiet)
        self.output_dir = None if output_dir is None else os.fspath(output_dir)
        self.trajectory_output_dir = (
            None if trajectory_output_dir is None else os.fspath(trajectory_output_dir)
        )
        self.collision_diagnostic_output_dir = (
            None
            if collision_diagnostic_output_dir is None
            else os.fspath(collision_diagnostic_output_dir)
        )
        self.timeout_diagnostic_output_dir = (
            None
            if timeout_diagnostic_output_dir is None
            else os.fspath(timeout_diagnostic_output_dir)
        )
        self.p_height_ablation = bool(p_height_ablation)
        self.p_height_gain = float(p_height_gain)
        self.z_q_scan_output_dir = (
            None if z_q_scan_output_dir is None else os.fspath(z_q_scan_output_dir)
        )
        self.action_interface_audit_output_dir = (
            None
            if action_interface_audit_output_dir is None
            else os.fspath(action_interface_audit_output_dir)
        )
        self.terminal_xy_q_scan_output_dir = (
            None
            if terminal_xy_q_scan_output_dir is None
            else os.fspath(terminal_xy_q_scan_output_dir)
        )
        self.terminal_coverage_output_dir = (
            None
            if terminal_coverage_output_dir is None
            else os.fspath(terminal_coverage_output_dir)
        )
        if self.p_height_ablation and self.p_height_gain <= 0.0:
            raise ValueError("p_height_gain must be positive when P-height ablation is enabled.")
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    def on_step(self, trainer, *, force=False):
        if not force and trainer.total_steps % self.interval != 0:
            return

        curriculum_metrics = {}
        if hasattr(trainer.env, "get_curriculum_metrics"):
            curriculum_metrics = dict(trainer.env.get_curriculum_metrics())
        if self.fixed_curriculum_stage is not None:
            curriculum_metrics["curriculum_stage"] = int(self.fixed_curriculum_stage)
        curriculum_metrics["replay_size"] = int(
            getattr(getattr(trainer, "buffer", None), "size", 0)
        )

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
        collision_windows = []
        timeout_records = []
        z_q_scan_records = []
        terminal_action_records = []
        terminal_xy_q_records = []
        terminal_coverage_records = []
        synthetic_action_mapping = []
        terminal_max_planar_speed = None

        if not self.quiet:
            print(f"[Eval @ {trainer.total_steps}] Starting evaluation...")

        with _suppress_native_output(self.quiet):
            eval_env = self.env_cls(**self.env_kwargs, gui=self.eval_gui)
            try:
                base_eval_env = getattr(eval_env, "base_env", eval_env)
                if self.eval_episode_len_sec is not None:
                    base_eval_env.EPISODE_LEN_SEC = self.eval_episode_len_sec
                terminal_max_planar_speed = float(eval_env.max_reference_speed)
                reward_model = getattr(base_eval_env, "_reward_model", None)
                terminal_guidance_radius = float(
                    getattr(reward_model, "near_goal_guidance_radius", 1.2)
                )
                terminal_goal_tolerance = float(eval_env.GOAL_TOLERANCE)
                if self.action_interface_audit_output_dir is not None:
                    for norm in (0.0, 0.1, 0.3, 0.5, 0.75, 1.0):
                        probe = np.array([norm, 0.0, 0.0], dtype=np.float32)
                        mapped = np.asarray(
                            eval_env._velocity_command(probe), dtype=np.float32
                        ).reshape(3)
                        synthetic_action_mapping.append(
                            {
                                "actor_action": probe.tolist(),
                                "actor_xy_norm": float(np.linalg.norm(probe[:2])),
                                "cmd_velocity": mapped.tolist(),
                                "cmd_xy_speed": float(np.linalg.norm(mapped[:2])),
                            }
                        )
                if self.fixed_curriculum_stage is not None:
                    eval_env.set_curriculum_stage_override(self.fixed_curriculum_stage)
                elif (
                    bool(getattr(eval_env, "CURRICULUM_ENABLED", False))
                    and hasattr(eval_env, "set_curriculum_stage_override")
                    and hasattr(trainer.env, "get_curriculum_stage")
                ):
                    eval_env.set_curriculum_stage_override(trainer.env.get_curriculum_stage())
                evaluation_seeds = (
                    self.episode_seeds if self.episode_seeds is not None
                    else tuple(self.episode_seed_offset + ep for ep in range(self.episodes))
                )
                for ep, environment_seed in enumerate(evaluation_seeds):
                    obs, reset_info = eval_env.reset(seed=environment_seed)
                    paired_trace = None
                    if self.paired_trace_output_dir is not None:
                        paired_trace = PairedTrajectoryTrace(eval_env, environment_seed)
                        paired_trace.install()
                    fixed_actions = None
                    fixed_commands = None
                    fixed_limit = None
                    if self.fixed_command_tape_dir is not None:
                        tape_path = os.path.join(self.fixed_command_tape_dir, f"trace_seed{environment_seed}.npz")
                        with np.load(tape_path) as tape:
                            fixed_actions = np.asarray(tape["upper_action"], dtype=np.float32).copy()
                            fixed_commands = np.asarray(tape["upper_command_velocity"], dtype=np.float32).copy()
                        if fixed_actions.ndim != 2 or fixed_commands.shape != (len(fixed_actions), 3):
                            raise ValueError(f"Invalid fixed command tape: {tape_path}")
                        if self.command_intervention_start_step is not None:
                            if self.command_intervention_start_step + self.command_intervention_steps > len(fixed_actions):
                                raise ValueError("Intervention window exceeds source command tape")
                            fixed_limit = None
                        else:
                            fixed_limit = len(fixed_actions) if self.fixed_command_steps is None else min(len(fixed_actions), self.fixed_command_steps)
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
                    ep_terminal_decisions = []
                    collision_history = deque(
                        maxlen=max(1, int(round(WINDOW_SECONDS / eval_env.CTRL_TIMESTEP)))
                    )
                    ep_positions = [
                        np.asarray(eval_env.pos[0], dtype=np.float32).copy()
                    ]
                    ep_velocities = [
                        np.asarray(eval_env.vel[0], dtype=np.float32).copy()
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
                            "policy_action": [],
                            "reward": [],
                            "goal_distance": [],
                            "min_tree_clearance": [],
                            "packet_version": [],
                            "packet_valid": [],
                            "cmd_velocity": [],
                            "delta_v": [],
                            "min_range_fraction": [],
                            "goal_error": [],
                        }
                        seed_zero_obstacle_specs = _trajectory_obstacle_specs(eval_env)
                        seed_zero_start = np.asarray(
                            getattr(eval_env, "START_POS", initial_position), dtype=np.float32
                        ).copy()
                        seed_zero_target = np.asarray(
                            getattr(eval_env, "TARGET_POS", initial_position), dtype=np.float32
                        ).copy()

                    while not done and (fixed_limit is None or step_idx < fixed_limit):
                        state_before_action = np.asarray(state, dtype=np.float32).copy()
                        policy_action = (
                            trainer.agent.select_action(state)
                            if fixed_actions is None or self.command_intervention_start_step is not None
                            else fixed_actions[step_idx].copy()
                        )
                        action = np.asarray(policy_action, dtype=np.float32).reshape(-1).copy()
                        position_before_action = np.asarray(
                            eval_env.pos[0], dtype=np.float32
                        ).reshape(3)
                        target_before_action = np.asarray(
                            getattr(eval_env, "TARGET_POS", (np.nan, np.nan, np.nan)),
                            dtype=np.float32,
                        ).reshape(3)
                        if self.z_q_scan_output_dir is not None:
                            if not hasattr(trainer.agent, "evaluate_critic_actions"):
                                raise RuntimeError(
                                    "Z-action Q scan requires an agent exposing "
                                    "evaluate_critic_actions()."
                                )
                            candidates = z_action_candidates(policy_action)
                            q1, q2 = trainer.agent.evaluate_critic_actions(state, candidates)
                            z_q_scan_records.append(
                                {
                                    "episode": int(ep),
                                    "seed": int(ep),
                                    "decision_index": int(step_idx),
                                    "time_seconds": float(step_idx) * float(eval_env.CTRL_TIMESTEP),
                                    "position": position_before_action.tolist(),
                                    "actual_velocity": np.asarray(
                                        eval_env.vel[0], dtype=np.float32
                                    ).reshape(3).tolist(),
                                    "goal_error": (target_before_action - position_before_action).tolist(),
                                    "goal_distance": float(
                                        np.linalg.norm(target_before_action - position_before_action)
                                    ),
                                    "policy_action": action.tolist(),
                                    "z_action_values": Z_ACTION_SCAN_VALUES.tolist(),
                                    "q1": q1.tolist(),
                                    "q2": q2.tolist(),
                                    "q_min": np.minimum(q1, q2).tolist(),
                                }
                            )
                        if self.p_height_ablation:
                            action, _ = p_height_ablation_action(
                                policy_action,
                                position=position_before_action,
                                target_position=target_before_action,
                                max_vertical_speed=float(eval_env.max_reference_vertical_speed),
                                gain=self.p_height_gain,
                            )
                        use_fixed_command = fixed_commands is not None and (
                            self.command_intervention_start_step is None
                            or self.command_intervention_start_step <= step_idx
                            < self.command_intervention_start_step + self.command_intervention_steps
                        )
                        if not use_fixed_command:
                            obs, reward, terminated, truncated, info = eval_env.step(action.reshape(1, -1))
                        else:
                            # Bypass the state-dependent height-hold mapping so
                            # Full receives the exact K1 three-dimensional command.
                            original_mapping = eval_env._velocity_command
                            fixed_command = fixed_commands[step_idx].copy()
                            eval_env._velocity_command = lambda _action, command=fixed_command: command.copy()
                            try:
                                obs, reward, terminated, truncated, info = eval_env.step(action.reshape(1, -1))
                            finally:
                                eval_env._velocity_command = original_mapping
                        if paired_trace is not None:
                            paired_trace.record_upper(
                                action, policy_action, reward, info or {},
                                pre_observation=state_before_action,
                            )
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
                        ep_velocities.append(
                            np.asarray(eval_env.vel[0], dtype=np.float32).copy()
                        )
                        ep_goal_distances.append(
                            float(last_info.get("goal_distance", np.nan))
                        )

                        command_velocity = np.asarray(
                            getattr(eval_env, "command_velocity", (np.nan, np.nan, np.nan)),
                            dtype=np.float32,
                        ).reshape(3)
                        actual_velocity = np.asarray(
                            getattr(eval_env, "vel", ((np.nan, np.nan, np.nan),))[0],
                            dtype=np.float32,
                        ).reshape(3)
                        position = np.asarray(
                            getattr(eval_env, "pos", ((np.nan, np.nan, np.nan),))[0],
                            dtype=np.float32,
                        ).reshape(3)
                        target = np.asarray(
                            getattr(eval_env, "TARGET_POS", (np.nan, np.nan, np.nan)),
                            dtype=np.float32,
                        ).reshape(3)
                        action_vector = np.zeros(3, dtype=np.float32)
                        flattened_action = np.asarray(action, dtype=np.float32).reshape(-1)
                        action_vector[: min(3, flattened_action.size)] = flattened_action[:3]
                        min_range_fraction = float(last_info.get("min_range_fraction", np.nan))
                        if not np.isfinite(min_range_fraction):
                            base_env = getattr(eval_env, "base_env", eval_env)
                            if hasattr(base_env, "_computeRangeObservation"):
                                min_range_fraction = float(
                                    np.min(base_env._computeRangeObservation(position))
                                )
                        collision_history.append(
                            {
                                "time_seconds": float(step_idx) * float(eval_env.CTRL_TIMESTEP),
                                "upper_action": action_vector.tolist(),
                                "cmd_velocity": command_velocity.tolist(),
                                "actual_velocity": actual_velocity.tolist(),
                                "delta_v": (command_velocity - actual_velocity).tolist(),
                                "clearance": float(last_info.get("min_tree_clearance", np.nan)),
                                "min_range_fraction": min_range_fraction,
                                "position": position.tolist(),
                                "goal_error": (target - position).tolist(),
                                "goal_distance": float(last_info.get("goal_distance", np.nan)),
                            }
                        )
                        if (
                            self.action_interface_audit_output_dir is not None
                            or self.terminal_xy_q_scan_output_dir is not None
                        ):
                            ep_terminal_decisions.append(
                                {
                                    "episode": int(ep),
                                    "seed": int(ep),
                                    "decision_index": int(step_idx - 1),
                                    "time_seconds": float(step_idx - 1)
                                    * float(eval_env.CTRL_TIMESTEP),
                                    "critic_state": state_before_action,
                                    "position": position_before_action.copy(),
                                    "target_position": target_before_action.copy(),
                                    "goal_distance_before": float(
                                        np.linalg.norm(target_before_action - position_before_action)
                                    ),
                                    "actor_action": np.asarray(
                                        policy_action, dtype=np.float32
                                    ).reshape(3).copy(),
                                    "cmd_velocity": command_velocity.copy(),
                                    "actual_velocity": actual_velocity.copy(),
                                    "clearance": float(
                                        last_info.get("min_tree_clearance", np.nan)
                                    ),
                                    "min_range_fraction": min_range_fraction,
                                }
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
                            seed_zero_trace["policy_action"].append(
                                np.asarray(policy_action, dtype=np.float32).copy()
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
                            seed_zero_trace["cmd_velocity"].append(command_velocity)
                            seed_zero_trace["delta_v"].append(
                                command_velocity - actual_velocity
                            )
                            seed_zero_trace["min_range_fraction"].append(
                                min_range_fraction
                            )
                            seed_zero_trace["goal_error"].append(target - position)

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

                    if fixed_limit is not None and not done:
                        last_info = {**last_info, "done_reason": "intervention_end"}
                    if paired_trace is not None:
                        paired_trace.uninstall()
                        trace_directory = self.paired_trace_output_dir
                        if self.paired_trace_step_dir:
                            trace_directory = os.path.join(
                                trace_directory, f"step_{int(trainer.total_steps):07d}"
                            )
                        paired_trace.save(
                            trace_directory,
                            done_reason=last_info.get("done_reason", "unknown"),
                            success=last_info.get("success", False),
                            final_goal_distance=last_info.get("goal_distance", np.nan),
                        )
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
                    if self.terminal_coverage_output_dir is not None:
                        coverage = summarize_terminal_episode(
                            positions=ep_positions,
                            velocities=ep_velocities,
                            target_position=getattr(eval_env, "TARGET_POS"),
                            goal_tolerance=terminal_goal_tolerance,
                            guidance_radius=terminal_guidance_radius,
                            done_reason=str(last_info.get("done_reason", "unknown")),
                            timestep_seconds=float(eval_env.CTRL_TIMESTEP),
                        )
                        terminal_coverage_records.append(
                            {
                                "episode": int(ep),
                                "seed": int(ep),
                                **coverage,
                            }
                        )
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
                    if bool(last_info.get("collision", False)):
                        collision_windows.append(
                            {
                                "episode": int(ep),
                                "seed": int(ep),
                                "length": int(step_idx),
                                "done_reason": str(last_info.get("done_reason", "collision")),
                                "classification": classify_collision_window(
                                    list(collision_history)
                                ),
                                "window_samples": list(collision_history),
                            }
                        )
                    if str(last_info.get("done_reason", "")) == "timeout":
                        timeout_classification = classify_timeout(
                            positions=ep_positions,
                            goal_distances=ep_goal_distances,
                            target_position=getattr(eval_env, "TARGET_POS"),
                            navigation_metrics=metrics,
                            last_window_samples=list(collision_history),
                        )
                        timeout_records.append(
                            {
                                "episode": int(ep),
                                "seed": int(ep),
                                "length": int(step_idx),
                                "classification": timeout_classification,
                            }
                        )
                        if timeout_classification["code"] == "T1":
                            for decision in ep_terminal_decisions:
                                actor_action = decision["actor_action"]
                                command = decision["cmd_velocity"]
                                actual = decision["actual_velocity"]
                                terminal_action_records.append(
                                    {
                                        key: (
                                            value.tolist()
                                            if isinstance(value, np.ndarray)
                                            else value
                                        )
                                        for key, value in decision.items()
                                        if key != "critic_state"
                                    }
                                    | {
                                        "actor_xy_norm": float(
                                            np.linalg.norm(actor_action[:2])
                                        ),
                                        "cmd_xy_speed": float(np.linalg.norm(command[:2])),
                                        "actual_xy_speed": float(np.linalg.norm(actual[:2])),
                                    }
                                )
                                if self.terminal_xy_q_scan_output_dir is None:
                                    continue
                                goal_distance = float(decision["goal_distance_before"])
                                clearance = float(decision["clearance"])
                                if not (
                                    0.3 < goal_distance < 1.2
                                    and np.isfinite(clearance)
                                    and clearance > 0.35
                                ):
                                    continue
                                if not hasattr(trainer.agent, "evaluate_critic_actions"):
                                    raise RuntimeError(
                                        "Terminal XY Q scan requires evaluate_critic_actions()."
                                    )
                                candidates, candidate_metadata = xy_action_candidates(
                                    actor_action,
                                    position=decision["position"],
                                    target_position=decision["target_position"],
                                    max_planar_speed=terminal_max_planar_speed,
                                )
                                q1, q2 = trainer.agent.evaluate_critic_actions(
                                    decision["critic_state"], candidates
                                )
                                terminal_xy_q_records.append(
                                    {
                                        "episode": int(ep),
                                        "seed": int(ep),
                                        "decision_index": int(decision["decision_index"]),
                                        "time_seconds": float(decision["time_seconds"]),
                                        "position": decision["position"].tolist(),
                                        "goal_distance": goal_distance,
                                        "clearance": clearance,
                                        "actor_action": actor_action.tolist(),
                                        "candidates": [
                                            {
                                                **metadata,
                                                "action": candidates[index].tolist(),
                                                "q1": float(q1[index]),
                                                "q2": float(q2[index]),
                                                "q_min": float(min(q1[index], q2[index])),
                                            }
                                            for index, metadata in enumerate(
                                                candidate_metadata
                                            )
                                        ],
                                    }
                                )
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
            if self.terminal_coverage_output_dir is not None:
                coverage_summary = summarize_terminal_coverage(terminal_coverage_records)
                for key in (
                    "guidance_entered_episode_fraction",
                    "guidance_state_fraction",
                    "successful_terminal_transition_count",
                    "minimum_goal_distance_m",
                ):
                    trainer.last_eval_info[f"eval/terminal_coverage/{key}"] = float(
                        coverage_summary[key]
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
                    "episode_seed_offset": self.episode_seed_offset,
                    "curriculum": curriculum_metrics,
                    "summary": {
                        key: None if value is None else float(value)
                        for key, value in trainer.last_eval_info.items()
                    },
                    "episodes": [
                        {
                            "episode": int(index),
                            "environment_seed": int(evaluation_seeds[index]),
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
                    seed=self.episode_seed_offset,
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
            if self.collision_diagnostic_output_dir is not None:
                os.makedirs(self.collision_diagnostic_output_dir, exist_ok=True)
                collision_record = {
                    "schema_version": "upper_collision_window_v1",
                    "training_step": int(trainer.total_steps),
                    "decision_rate_hz": float(1.0 / eval_env.CTRL_TIMESTEP),
                    "window_seconds": float(WINDOW_SECONDS),
                    "collision_count": int(len(collision_windows)),
                    "collisions": collision_windows,
                }
                latest_path = os.path.join(
                    self.collision_diagnostic_output_dir,
                    "collision_windows_latest.json",
                )
                with open(latest_path, "w", encoding="utf-8") as handle:
                    json.dump(collision_record, handle, ensure_ascii=False, indent=2)
                history_path = os.path.join(
                    self.collision_diagnostic_output_dir,
                    "collision_windows.jsonl",
                )
                with open(history_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(collision_record, ensure_ascii=False) + "\n")
                print(
                    f"[Collision windows @ {trainer.total_steps}] "
                    f"count={len(collision_windows)} | latest={latest_path}"
                )
            if self.timeout_diagnostic_output_dir is not None:
                os.makedirs(self.timeout_diagnostic_output_dir, exist_ok=True)
                timeout_record = {
                    "schema_version": "upper_timeout_diagnostic_v1",
                    "training_step": int(trainer.total_steps),
                    "decision_rate_hz": float(1.0 / eval_env.CTRL_TIMESTEP),
                    "window_seconds": float(WINDOW_SECONDS),
                    "timeout_count": int(len(timeout_records)),
                    "timeouts": timeout_records,
                }
                latest_path = os.path.join(
                    self.timeout_diagnostic_output_dir, "timeout_diagnostics_latest.json"
                )
                with open(latest_path, "w", encoding="utf-8") as handle:
                    json.dump(timeout_record, handle, ensure_ascii=False, indent=2)
                history_path = os.path.join(
                    self.timeout_diagnostic_output_dir, "timeout_diagnostics.jsonl"
                )
                with open(history_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(timeout_record, ensure_ascii=False) + "\n")
                print(
                    f"[Timeout diagnostics @ {trainer.total_steps}] "
                    f"count={len(timeout_records)} | latest={latest_path}"
                )
            if self.z_q_scan_output_dir is not None:
                os.makedirs(self.z_q_scan_output_dir, exist_ok=True)
                scan_record = {
                    "schema_version": "upper_z_action_critic_q_scan_v1",
                    "training_step": int(trainer.total_steps),
                    "decision_rate_hz": float(1.0 / eval_env.CTRL_TIMESTEP),
                    "z_action_values": Z_ACTION_SCAN_VALUES.tolist(),
                    "records": z_q_scan_records,
                }
                scan_path = os.path.join(
                    self.z_q_scan_output_dir, "z_action_critic_q_scan.json"
                )
                with open(scan_path, "w", encoding="utf-8") as handle:
                    json.dump(scan_record, handle, ensure_ascii=False, indent=2)
                print(
                    f"[Z-action Q scan @ {trainer.total_steps}] "
                    f"records={len(z_q_scan_records)} | path={scan_path}"
                )
            if self.action_interface_audit_output_dir is not None:
                exported = export_action_interface_audit(
                    self.action_interface_audit_output_dir,
                    records=terminal_action_records,
                    synthetic_mapping=synthetic_action_mapping,
                    max_planar_speed=terminal_max_planar_speed,
                )
                print(
                    f"[Action interface audit @ {trainer.total_steps}] "
                    f"records={len(terminal_action_records)} | path={exported['json']}"
                )
            if self.terminal_xy_q_scan_output_dir is not None:
                exported = export_xy_q_scan(
                    self.terminal_xy_q_scan_output_dir,
                    records=terminal_xy_q_records,
                )
                print(
                    f"[Terminal XY Q scan @ {trainer.total_steps}] "
                    f"states={len(terminal_xy_q_records)} | path={exported['json']}"
                )
            if self.terminal_coverage_output_dir is not None:
                exported = export_terminal_coverage(
                    self.terminal_coverage_output_dir,
                    episodes=terminal_coverage_records,
                    training_step=trainer.total_steps,
                    timestep_seconds=float(eval_env.CTRL_TIMESTEP),
                )
                summary = exported["summary"]
                print(
                    f"[Terminal coverage @ {trainer.total_steps}] "
                    f"entered={summary['guidance_entered_episode_count']}/"
                    f"{summary['episode_count']} | "
                    f"success_transitions="
                    f"{summary['successful_terminal_transition_count']} | "
                    f"path={exported['json']}"
                )

    def on_episode_end(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass
