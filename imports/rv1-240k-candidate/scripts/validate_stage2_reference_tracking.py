"""Actor-only Stage2-v1 gates using the frozen external ReferencePacket contract.

No teacher, RL update, planner, latent model, forest obstacle, or hierarchical
async cache is used.  This script only evaluates an already-created 30D Actor
checkpoint on hover and/or the same 7 m continuous trajectory as Stage 1.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.gym_pybullet_compat import ensure_gym_pybullet_envs_compat

ensure_gym_pybullet_envs_compat()

from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from algos.td3.td3_reference_tracking import (
    STAGE2_V1_CONTEXT_ID,
    TD3ReferenceTracking,
    stage2_v1_context_contract,
)
from envs.ForestAviary import CustomForestAviary
from envs.preprocess import preprocess_state
from models.continuous_world_reference import (
    ConstantWorldTrajectory,
    ContinuousWorldPacketSource,
    TrapezoidalWorldTrajectory,
    derive_motion_timing,
)
from models.motor_action_codec import ASYMMETRIC_RPM


@dataclass(frozen=True)
class Stage2ValidationConfig:
    start_position: tuple[float, float, float] = (-3.5, 0.0, 1.0)
    target_position: tuple[float, float, float] = (3.5, 0.0, 1.0)
    max_speed: float = 0.6
    max_acceleration: float = 1.5
    high_frequency_hz: int = 15
    low_frequency_hz: int = 120
    window_seconds: float = 1.0
    sequence_length: int = 16
    lookahead_seconds: float = 0.2
    settle_seconds: float = 2.0
    hover_duration_seconds: float = 30.0
    goal_tolerance: float = 0.20
    attitude_limit: float = 0.65

    def __post_init__(self):
        if self.low_frequency_hz % self.high_frequency_hz != 0:
            raise ValueError("Low frequency must be divisible by high frequency.")
        point_interval = self.window_seconds / float(self.sequence_length - 1)
        lookahead_points = self.lookahead_seconds / point_interval
        if not np.isclose(lookahead_points, round(lookahead_points)):
            raise ValueError("lookahead_seconds must align with a packet sample point.")

    @property
    def lookahead_points(self) -> int:
        point_interval = self.window_seconds / float(self.sequence_length - 1)
        return int(round(self.lookahead_seconds / point_interval))


def _quiet_env(**kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return CustomForestAviary(**kwargs)


def make_agent(env, config: Stage2ValidationConfig, device: torch.device):
    return TD3ReferenceTracking(
        state_dim=int(np.prod(env.observation_space.shape)),
        action_dim=4,
        max_action=1.0,
        target_position=config.target_position,
        ctrl_freq=config.low_frequency_hz,
        sequence_length=config.sequence_length,
        reference_horizon_seconds=config.window_seconds,
        high_level_interval=config.low_frequency_hz // config.high_frequency_hz,
        reference_mode="line",
        reference_source="external_packet",
        max_reference_speed=config.max_speed,
        max_reference_acceleration=config.max_acceleration,
        max_reference_vertical_speed=0.4,
        lookahead_points=config.lookahead_points,
        actor_structure="plain",
        lower_action_history_steps=0,
        motor_action_mode=ASYMMETRIC_RPM,
        device=device,
    )


def load_actor_checkpoint(agent: TD3ReferenceTracking, checkpoint_path: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
    if not isinstance(checkpoint, dict) or "actor_state_dict" not in checkpoint:
        raise ValueError(
            "Stage2-v1 expects a BC-style checkpoint containing actor_state_dict."
        )
    architecture = checkpoint.get("architecture", {})
    input_dim = int(architecture.get("input_dim", -1))
    output_dim = int(architecture.get("output_dim", -1))
    if input_dim != 30 or input_dim != agent.context_dim:
        raise ValueError(
            f"Stage2-v1 requires a 30D Actor checkpoint, got input_dim={input_dim}."
        )
    if output_dim != 4:
        raise ValueError(f"Stage2-v1 requires four motor outputs, got {output_dim}.")
    if int(architecture.get("multi_step_action_history", -1)) != 0:
        raise ValueError("Stage2-v1 checkpoints must not use multi-step action history.")
    if checkpoint.get("motor_action_codec") != ASYMMETRIC_RPM:
        raise ValueError("Stage2-v1 checkpoints must declare asymmetric_rpm encoding.")
    if checkpoint.get("context_contract_id") != STAGE2_V1_CONTEXT_ID:
        raise ValueError(
            "Checkpoint does not declare the frozen Stage2-v1 external-packet context."
        )
    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.actor.eval()
    return {
        "path": str(checkpoint_path),
        "architecture": architecture,
        "dataset_samples_sha256": checkpoint.get("dataset_samples_sha256"),
    }


def _task_setup(task: str, config: Stage2ValidationConfig):
    if task == "straight":
        trajectory = TrapezoidalWorldTrajectory(
            config.start_position,
            config.target_position,
            max_speed=config.max_speed,
            max_acceleration=config.max_acceleration,
        )
        timing = derive_motion_timing(
            trajectory,
            max_speed=config.max_speed,
            settle_seconds=config.settle_seconds,
        )
        return trajectory, float(timing["timeout"]), timing
    if task == "hover":
        trajectory = ConstantWorldTrajectory(
            config.start_position,
            duration=config.hover_duration_seconds,
        )
        timing = {
            "trajectory_duration": float(config.hover_duration_seconds),
            "timeout": float(config.hover_duration_seconds),
            "reachability_assertion": True,
        }
        return trajectory, float(config.hover_duration_seconds), timing
    raise ValueError(f"Unsupported Stage2-v1 task {task!r}.")


def run_episode(
    *,
    task: str,
    seed: int,
    checkpoint_path: str,
    config: Stage2ValidationConfig,
    device: torch.device,
) -> dict:
    trajectory, timeout, timing = _task_setup(task, config)
    env = _quiet_env(
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        initial_xyzs=np.asarray([config.start_position], dtype=np.float32),
        target_pos=np.asarray(config.target_position, dtype=np.float32),
        pyb_freq=240,
        ctrl_freq=config.low_frequency_hz,
        gui=False,
        curriculum=False,
        num_trees=0,
        route_blocking_tree=False,
        episode_len_sec=timeout,
        goal_tolerance=config.goal_tolerance,
    )
    try:
        observation, _ = env.reset(seed=int(seed))
        state = preprocess_state(observation.reshape(-1))
        agent = make_agent(env, config, device)
        checkpoint = load_actor_checkpoint(agent, checkpoint_path)
        agent.reset_episode()
        agent.configure_motor_action_interface(env)
        source = ContinuousWorldPacketSource(
            trajectory,
            window_seconds=config.window_seconds,
            sequence_length=config.sequence_length,
        )

        low_dt = 1.0 / config.low_frequency_hz
        high_interval = config.low_frequency_hz // config.high_frequency_hz
        position_squared_errors = []
        velocity_squared_errors = []
        height_abs_errors = []
        angular_velocity_squared = []
        action_delta_squared = []
        packet_ages = []
        accepted_versions = []
        max_roll = max_pitch = 0.0
        attitude_violation_count = 0
        saturated_motor_values = motor_value_count = 0
        missing_packet_count = stale_packet_count = fallback_count = 0
        teacher_intervention_count = 0
        step_count = 0
        info = {}
        terminated = truncated = False
        previous_action = np.zeros(4, dtype=np.float32)

        while not (terminated or truncated):
            sim_time = step_count * low_dt
            if step_count % high_interval == 0:
                packet = source.make_packet(sim_time)
                if agent.publish_reference(packet):
                    accepted_versions.append(int(packet.version))

            context = agent.prepare_runtime_context(state)
            sample = agent.last_reference_sample
            if sample is None:
                missing_packet_count += 1
            elif not sample["valid"]:
                stale_packet_count += 1
            fallback = not bool(agent.last_runtime_info.get("async_cache_valid", 0.0))
            fallback_count += int(fallback)

            action = agent.action_from_context(context)
            # This evaluator intentionally has no DSLPID object or teacher call.
            teacher_intervention_count += 0
            action = np.asarray(action, dtype=np.float32).reshape(4)
            saturated_motor_values += int(np.sum(np.abs(action) >= 1.0 - 1e-6))
            motor_value_count += 4
            action_delta_squared.append(float(np.mean(np.square(action - previous_action))))
            previous_action = action.copy()
            agent.record_executed_action(action)

            next_observation, _, terminated, truncated, info = env.step(
                action.reshape(1, -1)
            )
            next_state = preprocess_state(next_observation.reshape(-1))
            actual_position = np.asarray(env.pos[0], dtype=np.float64)
            actual_velocity = np.asarray(env.vel[0], dtype=np.float64)
            if sample is not None:
                reference_position = np.asarray(sample["current_position"], dtype=np.float64)
                reference_velocity = np.asarray(sample["current_velocity"], dtype=np.float64)
                position_squared_errors.append(
                    float(np.sum(np.square(actual_position - reference_position)))
                )
                velocity_squared_errors.append(
                    float(np.sum(np.square(actual_velocity - reference_velocity)))
                )
                height_abs_errors.append(
                    abs(float(actual_position[2] - reference_position[2]))
                )
                packet_ages.append(float(sample["age_seconds"]))
            roll = abs(float(env.rpy[0][0]))
            pitch = abs(float(env.rpy[0][1]))
            max_roll = max(max_roll, roll)
            max_pitch = max(max_pitch, pitch)
            attitude_violation_count += int(
                roll > config.attitude_limit or pitch > config.attitude_limit
            )
            angular_velocity_squared.append(
                float(np.sum(np.square(np.asarray(env.ang_v[0], dtype=np.float64))))
            )

            agent.advance_runtime_step()
            state = next_state
            step_count += 1

        position_values = np.asarray(position_squared_errors, dtype=np.float64)
        velocity_values = np.asarray(velocity_squared_errors, dtype=np.float64)
        height_values = np.asarray(height_abs_errors, dtype=np.float64)
        age_values = np.asarray(packet_ages, dtype=np.float64)
        versions_continuous = accepted_versions == list(
            range(1, len(accepted_versions) + 1)
        )
        final_distance = float(
            np.linalg.norm(
                np.asarray(config.target_position, dtype=np.float64)
                - np.asarray(env.pos[0], dtype=np.float64)
            )
        )
        if task == "straight":
            task_success = bool(info.get("success", False))
        else:
            task_success = bool(
                info.get("done_reason") == "timeout"
                and attitude_violation_count == 0
            )
        return {
            "task": task,
            "seed": int(seed),
            "success": task_success,
            "environment_success": bool(info.get("success", False)),
            "done_reason": str(info.get("done_reason", "unknown")),
            "steps": int(step_count),
            "final_distance": final_distance,
            "position_rmse": float(np.sqrt(position_values.mean())),
            "velocity_tracking_error": float(np.sqrt(velocity_values.mean())),
            "max_roll": float(max_roll),
            "max_pitch": float(max_pitch),
            "attitude_violation_count": int(attitude_violation_count),
            "height_abs_error": float(height_values.mean()),
            "height_rmse": float(np.sqrt(np.mean(np.square(height_values)))),
            "angular_velocity_rms": float(np.sqrt(np.mean(angular_velocity_squared))),
            "action_delta_rms": float(np.sqrt(np.mean(action_delta_squared))),
            "motor_saturation_ratio": float(
                saturated_motor_values / max(1, motor_value_count)
            ),
            "packet_age_mean": float(age_values.mean()),
            "packet_age_p95": float(np.percentile(age_values, 95.0)),
            "packet_age_max": float(age_values.max()),
            "missing_packet_count": int(missing_packet_count),
            "stale_packet_count": int(stale_packet_count),
            "fallback_count": int(fallback_count),
            "packet_version_continuity": bool(versions_continuous),
            "teacher_intervention_count": int(teacher_intervention_count),
            "actor_control_ratio": 1.0,
            "accepted_packet_count": len(accepted_versions),
            "checkpoint": checkpoint,
            **timing,
        }
    finally:
        env.close()


def task_gate(task: str, episodes: list[dict]) -> dict:
    position_limit = 0.05 if task == "hover" else 0.10
    height_limit = 0.03 if task == "hover" else 0.05
    checks = {
        "success_rate": all(episode["success"] for episode in episodes),
        "final_distance": (
            True
            if task == "hover"
            else all(episode["final_distance"] <= 0.20 for episode in episodes)
        ),
        "position_rmse": all(
            episode["position_rmse"] <= position_limit for episode in episodes
        ),
        "velocity_tracking_error": all(
            episode["velocity_tracking_error"] <= 0.15 for episode in episodes
        ),
        "attitude": all(
            episode["max_roll"] <= 0.65
            and episode["max_pitch"] <= 0.65
            and episode["attitude_violation_count"] == 0
            for episode in episodes
        ),
        "height": all(
            episode["height_rmse"] <= height_limit for episode in episodes
        ),
        "motor_saturation": all(
            episode["motor_saturation_ratio"] <= 0.05 for episode in episodes
        ),
        "packets": all(
            episode["missing_packet_count"] == 0
            and episode["stale_packet_count"] == 0
            and episode["fallback_count"] == 0
            and episode["packet_version_continuity"]
            for episode in episodes
        ),
        "actor_only": all(
            episode["teacher_intervention_count"] == 0
            and episode["actor_control_ratio"] == 1.0
            for episode in episodes
        ),
    }
    return {
        "task": task,
        "episode_count": len(episodes),
        "success_rate": float(np.mean([episode["success"] for episode in episodes])),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", nargs="+", choices=("hover", "straight"), default=["hover", "straight"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--output",
        default="analysis_outputs/stage2_reference_tracking/interface_gate_results.json",
    )
    args = parser.parse_args()
    if len(args.seeds) < 3:
        raise ValueError("Stage2-v1 validation requires at least three fixed seeds.")
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)

    config = Stage2ValidationConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    episodes = [
        run_episode(
            task=task,
            seed=seed,
            checkpoint_path=args.checkpoint,
            config=config,
            device=device,
        )
        for task in args.tasks
        for seed in args.seeds
    ]
    gates = {
        task: task_gate(
            task,
            [episode for episode in episodes if episode["task"] == task],
        )
        for task in args.tasks
    }
    result = {
        "contract": "stage2_v1_external_packet_actor_only",
        "config": asdict(config),
        "context_dim": 30,
        "context_contract_id": STAGE2_V1_CONTEXT_ID,
        "context_contract": stage2_v1_context_contract(),
        "reference_packet": "models.reference_packet.ReferencePacket",
        "reference_source": "external_packet",
        "motor_action_mode": ASYMMETRIC_RPM,
        "teacher_intervention_count": 0,
        "rl_training_performed": False,
        "upper_planner_used": False,
        "latent_used": False,
        "forest_obstacles": 0,
        "episodes": episodes,
        "gates": gates,
        "stage2_v1_gate_passed": bool(all(gate["passed"] for gate in gates.values())),
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
