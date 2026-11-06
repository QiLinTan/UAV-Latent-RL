"""Collect Stage2-v1 external-ReferencePacket DSLPID datasets.

This collector is intentionally separate from the historical BC collector.  It
uses the frozen Stage 1 trajectory/packet source and the frozen Stage2-v1 30D
context, and labels the current reference at ``sim_time + dt``.  The formal
profile uses the frozen canonical world-angular-velocity DSLPID teacher.  It
never runs an Actor or performs a network update.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pybullet as p


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.gym_pybullet_compat import ensure_gym_pybullet_envs_compat

ensure_gym_pybullet_envs_compat()

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from algos.td3.td3_reference_tracking import (
    STAGE2_V1_CONTEXT_ID,
    TD3ReferenceTracking,
    stage2_v1_context_contract,
)
from envs.ForestAviary import CustomForestAviary
from envs.preprocess import preprocess_state
from models.canonical_dslpid_teacher import (
    CANONICAL_DSLPID_WORLD_RATE_ID,
    CanonicalDSLPIDTeacher,
    WORLD_ANGULAR_VELOCITY,
    previous_rpy_from_world_angular_velocity,
)
from models.continuous_world_reference import (
    ConstantWorldTrajectory,
    ContinuousWorldPacketSource,
    TrapezoidalWorldTrajectory,
    derive_motion_timing,
)
from models.motor_action_codec import ASYMMETRIC_RPM


PILOT_DATASET_VERSION = "stage2_v1_external_packet_pilot"
FORMAL_DATASET_VERSION = "stage2_v1_external_packet_canonical_v2"
DATASET_VERSION = PILOT_DATASET_VERSION
STATEFUL_TEACHER = "stateful_dslpid"
CANONICAL_WORLD_RATE_TEACHER = "canonical_world_angular_velocity"
FORMAL_SEEDS = tuple(range(51))
TASK_IDS = {"hover": 0, "straight": 1}


@dataclass(frozen=True)
class Stage2PilotConfig:
    start_position: tuple[float, float, float] = (-3.5, 0.0, 1.0)
    target_position: tuple[float, float, float] = (3.5, 0.0, 1.0)
    max_speed: float = 0.6
    max_acceleration: float = 1.5
    packet_frequency_hz: int = 15
    control_frequency_hz: int = 120
    physics_frequency_hz: int = 240
    window_seconds: float = 1.0
    sequence_length: int = 16
    lookahead_seconds: float = 0.2
    settle_seconds: float = 2.0
    hover_duration_seconds: float = 30.0
    goal_tolerance: float = 0.2
    attitude_limit: float = 0.65

    def __post_init__(self):
        if self.control_frequency_hz % self.packet_frequency_hz != 0:
            raise ValueError("Control frequency must be divisible by packet frequency.")
        point_dt = self.window_seconds / float(self.sequence_length - 1)
        points = self.lookahead_seconds / point_dt
        if not np.isclose(points, round(points)):
            raise ValueError("lookahead_seconds must align with a packet sample.")

    @property
    def packet_interval_steps(self) -> int:
        return self.control_frequency_hz // self.packet_frequency_hz

    @property
    def lookahead_points(self) -> int:
        point_dt = self.window_seconds / float(self.sequence_length - 1)
        return int(round(self.lookahead_seconds / point_dt))


@dataclass(frozen=True)
class PilotDisturbance:
    name: str
    initial_rpy: tuple[float, float, float]
    initial_velocity: tuple[float, float, float]
    initial_angular_velocity: tuple[float, float, float]
    height_offset: float


def disturbance_for_seed(seed: int) -> PilotDisturbance:
    """Return small deterministic perturbations without using seed as hidden noise."""

    profiles = {
        0: PilotDisturbance(
            "nominal",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            0.0,
        ),
        1: PilotDisturbance(
            "small_positive_recovery",
            (0.04, -0.03, 0.02),
            (0.12, -0.08, 0.0),
            (0.12, -0.10, 0.06),
            0.04,
        ),
        2: PilotDisturbance(
            "small_negative_recovery",
            (-0.04, 0.03, -0.02),
            (-0.12, 0.08, 0.0),
            (-0.12, 0.10, -0.06),
            -0.04,
        ),
    }
    if int(seed) not in profiles:
        raise ValueError("The pilot contract defines exactly seeds 0, 1, and 2.")
    return profiles[int(seed)]


def formal_disturbance_for_seed(seed: int) -> PilotDisturbance:
    """Return a deterministic, paired perturbation in the verified envelope."""

    seed = int(seed)
    if seed not in FORMAL_SEEDS:
        raise ValueError(f"Formal Stage2-v1 seed must be in {FORMAL_SEEDS[0]}..{FORMAL_SEEDS[-1]}.")
    if seed == 0:
        return disturbance_for_seed(0)
    pair_id = (seed + 1) // 2
    sign = 1.0 if seed % 2 else -1.0
    rng = np.random.default_rng(0x5A17 + pair_id)
    base_rpy = rng.uniform(
        low=[0.008, 0.008, 0.0],
        high=[0.04, 0.04, 0.03],
    ) * rng.choice([-1.0, 1.0], size=3)
    base_velocity = rng.uniform(
        low=[0.02, 0.02, 0.0],
        high=[0.12, 0.12, 0.05],
    ) * rng.choice([-1.0, 1.0], size=3)
    base_angular_velocity = rng.uniform(
        low=[0.02, 0.02, 0.0],
        high=[0.12, 0.12, 0.08],
    ) * rng.choice([-1.0, 1.0], size=3)
    height_offset = float(rng.uniform(0.008, 0.04) * rng.choice([-1.0, 1.0]))
    prefix = "positive" if sign > 0 else "negative"
    return PilotDisturbance(
        f"paired_{prefix}_{pair_id:03d}",
        tuple((sign * base_rpy).tolist()),
        tuple((sign * base_velocity).tolist()),
        tuple((sign * base_angular_velocity).tolist()),
        sign * height_offset,
    )


def formal_split_for_seed(seed: int) -> str:
    seed = int(seed)
    if seed not in FORMAL_SEEDS:
        raise ValueError(f"Unknown formal seed {seed}.")
    if seed <= 38:
        return "train"
    if seed <= 44:
        return "validation"
    return "test"


def _quiet_env(**kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return CustomForestAviary(**kwargs)


def make_env(
    task: str,
    disturbance: PilotDisturbance,
    config: Stage2PilotConfig,
    timeout: float,
):
    initial_position = np.asarray(config.start_position, dtype=np.float32).copy()
    initial_position[2] += float(disturbance.height_offset)
    return _quiet_env(
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        initial_xyzs=initial_position.reshape(1, 3),
        initial_rpys=np.asarray([disturbance.initial_rpy], dtype=np.float32),
        target_pos=np.asarray(config.target_position, dtype=np.float32),
        pyb_freq=config.physics_frequency_hz,
        ctrl_freq=config.control_frequency_hz,
        gui=False,
        curriculum=False,
        num_trees=0,
        route_blocking_tree=False,
        episode_len_sec=float(timeout),
        goal_tolerance=config.goal_tolerance,
    )


def make_agent(env, config: Stage2PilotConfig) -> TD3ReferenceTracking:
    return TD3ReferenceTracking(
        state_dim=int(np.prod(env.observation_space.shape)),
        action_dim=4,
        max_action=1.0,
        target_position=config.target_position,
        ctrl_freq=config.control_frequency_hz,
        sequence_length=config.sequence_length,
        reference_horizon_seconds=config.window_seconds,
        high_level_interval=config.packet_interval_steps,
        reference_mode="line",
        reference_source="external_packet",
        max_reference_speed=config.max_speed,
        max_reference_acceleration=config.max_acceleration,
        max_reference_vertical_speed=0.4,
        lookahead_points=config.lookahead_points,
        actor_structure="plain",
        lower_action_history_steps=0,
        motor_action_mode=ASYMMETRIC_RPM,
    )


def make_task_trajectory(task: str, config: Stage2PilotConfig):
    if task == "hover":
        trajectory = ConstantWorldTrajectory(
            config.start_position,
            duration=config.hover_duration_seconds,
        )
        return trajectory, config.hover_duration_seconds, {
            "trajectory_duration": config.hover_duration_seconds,
            "timeout": config.hover_duration_seconds,
            "reachability_assertion": True,
        }
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
    raise ValueError(f"Unsupported Stage2-v1 task {task!r}.")


def physical_state(env) -> np.ndarray:
    state = env._getDroneStateVector(0)
    return np.concatenate(
        [state[0:3], state[7:10], state[10:13], state[13:16]]
    ).astype(np.float32)


def apply_initial_disturbance(env, disturbance: PilotDisturbance) -> None:
    linear = np.asarray(disturbance.initial_velocity, dtype=np.float64)
    angular = np.asarray(disturbance.initial_angular_velocity, dtype=np.float64)
    if np.any(linear != 0.0) or np.any(angular != 0.0):
        p.resetBaseVelocity(
            env.DRONE_IDS[0],
            linearVelocity=linear,
            angularVelocity=angular,
            physicsClientId=env.CLIENT,
        )
        env._updateAndStoreKinematicInformation()


def teacher_hidden_state(controller: DSLPIDControl) -> dict[str, np.ndarray | int]:
    return {
        "integral_pos_e": np.asarray(controller.integral_pos_e, dtype=np.float32).copy(),
        "integral_rpy_e": np.asarray(controller.integral_rpy_e, dtype=np.float32).copy(),
        "last_rpy": np.asarray(controller.last_rpy, dtype=np.float32).copy(),
        "last_rpy_e": np.asarray(controller.last_rpy_e, dtype=np.float32).copy(),
        "control_counter": int(controller.control_counter),
    }


def motor_wrench(rpm, *, kf: float, km: float, arm_length: float) -> np.ndarray:
    rpm = np.asarray(rpm, dtype=np.float64)
    forces = float(kf) * np.square(rpm)
    squared = np.square(rpm)
    return np.array(
        [
            np.sum(forces),
            (forces[0] + forces[1] - forces[2] - forces[3])
            * arm_length
            / np.sqrt(2.0),
            (-forces[0] + forces[1] + forces[2] - forces[3])
            * arm_length
            / np.sqrt(2.0),
            (-squared[0] + squared[1] - squared[2] + squared[3]) * km,
        ],
        dtype=np.float32,
    )


def reconstruct_context(
    state: np.ndarray,
    current_reference: np.ndarray,
    future_reference: np.ndarray,
    age_ratio: float,
    valid: bool,
    previous_action: np.ndarray,
) -> np.ndarray:
    mean = TD3ReferenceTracking._KIN_MEAN
    scale = TD3ReferenceTracking._KIN_STD
    state = np.asarray(state, dtype=np.float32).reshape(12)
    current_reference = np.asarray(current_reference, dtype=np.float32).reshape(6)
    future_reference = np.asarray(future_reference, dtype=np.float32).reshape(6)
    position = state[:3]
    velocity = state[6:9]
    return np.concatenate(
        [
            (state - mean) / scale,
            np.clip((current_reference[:3] - position) / scale[:3], -4.0, 4.0),
            np.clip((current_reference[3:] - velocity) / scale[6:9], -4.0, 4.0),
            np.clip((future_reference[:3] - position) / scale[:3], -4.0, 4.0),
            np.clip((future_reference[3:] - velocity) / scale[6:9], -4.0, 4.0),
            np.asarray([age_ratio, float(valid)], dtype=np.float32),
            np.asarray(previous_action, dtype=np.float32).reshape(4),
        ]
    ).astype(np.float32)


def _new_arrays() -> dict[str, list]:
    names = (
        "episode_id",
        "task_id",
        "disturbance_profile_id",
        "seed",
        "step_id",
        "sim_time",
        "control_time",
        "state",
        "state_quaternion",
        "context",
        "current_reference",
        "future_reference",
        "packet_version",
        "packet_t_gen",
        "packet_age",
        "packet_age_ratio",
        "packet_valid",
        "previous_action",
        "raw_teacher_rpm",
        "encoded_teacher_action",
        "applied_action",
        "next_state",
        "motor_saturation_mask",
        "teacher_wrench",
        "teacher_integral_pos_e_before",
        "teacher_integral_rpy_e_before",
        "teacher_last_rpy_before",
        "teacher_last_rpy_e_before",
        "teacher_control_counter_before",
        "teacher_integral_pos_e_after",
        "teacher_integral_rpy_e_after",
        "teacher_last_rpy_after",
        "teacher_last_rpy_e_after",
        "teacher_control_counter_after",
    )
    return {name: [] for name in names}


def _append_hidden(arrays: dict[str, list], suffix: str, hidden: dict) -> None:
    for name in ("integral_pos_e", "integral_rpy_e", "last_rpy", "last_rpy_e"):
        arrays[f"teacher_{name}_{suffix}"].append(hidden[name])
    arrays[f"teacher_control_counter_{suffix}"].append(hidden["control_counter"])


def collect_episode(
    *,
    episode_id: int,
    task: str,
    seed: int,
    disturbance_profile_id: int,
    disturbance: PilotDisturbance,
    teacher_mode: str,
    config: Stage2PilotConfig,
    arrays: dict[str, list],
) -> tuple[dict, dict]:
    trajectory, timeout, timing = make_task_trajectory(task, config)
    env = make_env(task, disturbance, config, timeout)
    try:
        observation, _ = env.reset(seed=int(seed))
        apply_initial_disturbance(env, disturbance)
        observation = env._computeObs()
        state = preprocess_state(observation.reshape(-1))
        agent = make_agent(env, config)
        agent.reset_episode()
        agent.set_high_level_enabled(False)
        codec = agent.configure_motor_action_interface(env)
        if teacher_mode == STATEFUL_TEACHER:
            teacher = DSLPIDControl(drone_model=env.DRONE_MODEL)
        elif teacher_mode == CANONICAL_WORLD_RATE_TEACHER:
            teacher = CanonicalDSLPIDTeacher(
                env.DRONE_MODEL,
                rate_mode=WORLD_ANGULAR_VELOCITY,
            )
        else:
            raise ValueError(f"Unsupported Stage2-v1 teacher mode {teacher_mode!r}.")
        teacher.reset()
        source = ContinuousWorldPacketSource(
            trajectory,
            window_seconds=config.window_seconds,
            sequence_length=config.sequence_length,
        )

        low_dt = 1.0 / config.control_frequency_hz
        position_squared_errors = []
        velocity_squared_errors = []
        height_squared_errors = []
        accepted_versions = []
        saturation_count = 0
        action_value_count = 0
        max_context_reconstruction_error = 0.0
        max_rpm_roundtrip_error = 0.0
        max_roll = 0.0
        max_pitch = 0.0
        max_canonical_hidden_contract_error = 0.0
        fallback_count = 0
        missing_packet_count = 0
        stale_packet_count = 0
        step = 0
        terminated = truncated = False
        info = {}

        while not (terminated or truncated):
            sim_time = step * low_dt
            if step % config.packet_interval_steps == 0:
                packet = source.make_packet(sim_time)
                if not agent.publish_reference(packet):
                    raise RuntimeError("Stage2-v1 packet publication was rejected.")
                accepted_versions.append(int(packet.version))

            context = agent.prepare_runtime_context(state)
            reference = agent.last_reference_sample
            if reference is None:
                missing_packet_count += 1
                raise RuntimeError("Stage2-v1 pilot encountered a missing packet.")
            if not bool(reference["valid"]):
                stale_packet_count += 1
                raise RuntimeError("Stage2-v1 pilot encountered a stale packet.")
            fallback_count += int(
                bool(agent.last_runtime_info.get("async_degraded_fallback", 0.0))
            )
            control_time = float(agent.last_runtime_info["reference_sample_time_seconds"])
            expected_control_time = sim_time + low_dt
            if not np.isclose(control_time, expected_control_time, atol=1e-12):
                raise AssertionError("Stage2-v1 control_time is not sim_time + dt.")

            current_reference = np.concatenate(
                [reference["current_position"], reference["current_velocity"]]
            ).astype(np.float32)
            future_reference = np.concatenate(
                [reference["lookahead_position"], reference["lookahead_velocity"]]
            ).astype(np.float32)
            current_state = physical_state(env)
            previous_action = agent.previous_action.copy()
            reconstructed = reconstruct_context(
                current_state,
                current_reference,
                future_reference,
                float(reference["age_ratio"]),
                bool(reference["valid"]),
                previous_action,
            )
            context_error = float(np.max(np.abs(context - reconstructed)))
            max_context_reconstruction_error = max(
                max_context_reconstruction_error,
                context_error,
            )
            if context.shape != (30,) or context_error > 1e-6:
                raise AssertionError(
                    f"Frozen 30D context reconstruction failed: max_error={context_error}."
                )

            drone_state = env._getDroneStateVector(0)
            state_quaternion = np.asarray(drone_state[3:7], dtype=np.float32).copy()
            if teacher_mode == STATEFUL_TEACHER:
                hidden_before = teacher_hidden_state(teacher)
            raw_rpm, _, _ = teacher.computeControl(
                control_timestep=env.CTRL_TIMESTEP,
                cur_pos=drone_state[0:3],
                cur_quat=drone_state[3:7],
                cur_vel=drone_state[10:13],
                cur_ang_vel=drone_state[13:16],
                target_pos=reference["current_position"],
                target_vel=reference["current_velocity"],
            )
            if teacher_mode == STATEFUL_TEACHER:
                hidden_after = teacher_hidden_state(teacher)
            else:
                hidden_before = teacher.last_hidden_before
                hidden_after = teacher.last_hidden_after
                if hidden_before is None or hidden_after is None:
                    raise AssertionError("Canonical teacher did not expose hidden-state snapshots.")
                expected_last_rpy = previous_rpy_from_world_angular_velocity(
                    drone_state[3:7],
                    drone_state[13:16],
                    env.CTRL_TIMESTEP,
                )
                hidden_errors = (
                    np.max(np.abs(hidden_before["integral_pos_e"])),
                    np.max(np.abs(hidden_before["integral_rpy_e"])),
                    np.max(np.abs(hidden_before["last_rpy_e"])),
                    abs(int(hidden_before["control_counter"])),
                    np.max(np.abs(hidden_before["last_rpy"] - expected_last_rpy)),
                )
                canonical_error = float(max(hidden_errors))
                max_canonical_hidden_contract_error = max(
                    max_canonical_hidden_contract_error,
                    canonical_error,
                )
                if canonical_error > 1e-6:
                    raise AssertionError(
                        "Canonical teacher hidden-state contract failed: "
                        f"max_error={canonical_error}."
                    )
            encoded_action = codec.rpm_to_normalized_action(raw_rpm).astype(np.float32)
            applied_action = encoded_action.copy()
            if not np.array_equal(encoded_action, applied_action):
                raise AssertionError("Encoded teacher action must equal applied action.")
            reconstructed_rpm = codec.normalized_action_to_rpm(applied_action)
            rpm_error = float(np.max(np.abs(reconstructed_rpm - raw_rpm)))
            max_rpm_roundtrip_error = max(max_rpm_roundtrip_error, rpm_error)
            saturation_mask = codec.physical_constraint.saturation_mask(raw_rpm, atol=1e-6)
            if not np.all(saturation_mask) and rpm_error > 1e-2:
                raise AssertionError(
                    f"Unsaturated RPM round-trip error exceeded 0.01 RPM: {rpm_error}."
                )
            saturation_count += int(np.sum(saturation_mask))
            action_value_count += 4
            wrench = motor_wrench(raw_rpm, kf=env.KF, km=env.KM, arm_length=env.L)

            next_observation, _, terminated, truncated, info = env.step(
                applied_action.reshape(1, -1)
            )
            next_state = physical_state(env)
            next_preprocessed_state = preprocess_state(next_observation.reshape(-1))

            arrays["episode_id"].append(episode_id)
            arrays["task_id"].append(TASK_IDS[task])
            arrays["disturbance_profile_id"].append(disturbance_profile_id)
            arrays["seed"].append(seed)
            arrays["step_id"].append(step)
            arrays["sim_time"].append(sim_time)
            arrays["control_time"].append(control_time)
            arrays["state"].append(current_state)
            arrays["state_quaternion"].append(state_quaternion)
            arrays["context"].append(context)
            arrays["current_reference"].append(current_reference)
            arrays["future_reference"].append(future_reference)
            arrays["packet_version"].append(reference["version"])
            arrays["packet_t_gen"].append(agent.reference_buffer.packet.t_gen)
            arrays["packet_age"].append(reference["age_seconds"])
            arrays["packet_age_ratio"].append(reference["age_ratio"])
            arrays["packet_valid"].append(reference["valid"])
            arrays["previous_action"].append(previous_action)
            arrays["raw_teacher_rpm"].append(raw_rpm)
            arrays["encoded_teacher_action"].append(encoded_action)
            arrays["applied_action"].append(applied_action)
            arrays["next_state"].append(next_state)
            arrays["motor_saturation_mask"].append(saturation_mask)
            arrays["teacher_wrench"].append(wrench)
            _append_hidden(arrays, "before", hidden_before)
            _append_hidden(arrays, "after", hidden_after)

            actual_position = np.asarray(env.pos[0], dtype=np.float64)
            actual_velocity = np.asarray(env.vel[0], dtype=np.float64)
            position_squared_errors.append(
                float(np.sum(np.square(actual_position - current_reference[:3])))
            )
            velocity_squared_errors.append(
                float(np.sum(np.square(actual_velocity - current_reference[3:])))
            )
            height_squared_errors.append(
                float(np.square(actual_position[2] - current_reference[2]))
            )
            max_roll = max(max_roll, abs(float(env.rpy[0][0])))
            max_pitch = max(max_pitch, abs(float(env.rpy[0][1])))

            agent.record_executed_action(applied_action)
            agent.advance_runtime_step()
            state = next_preprocessed_state
            step += 1

        expected_versions = list(range(1, len(accepted_versions) + 1))
        final_distance = float(
            np.linalg.norm(
                np.asarray(config.target_position, dtype=np.float64)
                - np.asarray(env.pos[0], dtype=np.float64)
            )
        )
        position_rmse = float(np.sqrt(np.mean(position_squared_errors)))
        velocity_error = float(np.sqrt(np.mean(velocity_squared_errors)))
        height_rmse = float(np.sqrt(np.mean(height_squared_errors)))
        done_reason = str(info.get("done_reason", "unknown"))
        if task == "hover":
            success = done_reason == "timeout"
            position_limit = 0.05
            height_limit = 0.03
            final_distance_check = True
        else:
            success = bool(info.get("success", False))
            position_limit = 0.10
            height_limit = 0.05
            final_distance_check = final_distance <= 0.20
        gate_checks = {
            "success": bool(success),
            "final_distance": bool(final_distance_check),
            "position_rmse": position_rmse <= position_limit,
            "velocity_tracking_error": velocity_error <= 0.15,
            "attitude": max_roll <= config.attitude_limit
            and max_pitch <= config.attitude_limit,
            "height": height_rmse <= height_limit,
            "motor_saturation": saturation_count / max(1, action_value_count) <= 0.05,
            "packets": missing_packet_count == 0
            and stale_packet_count == 0
            and fallback_count == 0
            and accepted_versions == expected_versions,
            "teacher_only": True,
            "context_reconstruction": max_context_reconstruction_error <= 1e-6,
        }
        episode = {
            "episode_id": episode_id,
            "task": task,
            "seed": seed,
            "disturbance": asdict(disturbance),
            "steps": step,
            "done_reason": done_reason,
            "success": bool(success),
            "final_distance": final_distance,
            "position_rmse": position_rmse,
            "velocity_tracking_error": velocity_error,
            "height_rmse": height_rmse,
            "max_roll": max_roll,
            "max_pitch": max_pitch,
            "motor_saturation_ratio": saturation_count / max(1, action_value_count),
            "missing_packet_count": missing_packet_count,
            "stale_packet_count": stale_packet_count,
            "fallback_count": fallback_count,
            "packet_version_continuity": accepted_versions == expected_versions,
            "accepted_packet_count": len(accepted_versions),
            "max_context_reconstruction_error": max_context_reconstruction_error,
            "max_rpm_roundtrip_error": max_rpm_roundtrip_error,
            "max_canonical_hidden_contract_error": (
                max_canonical_hidden_contract_error
            ),
            "teacher_intervention_count": step,
            "actor_control_count": 0,
            "gate_checks": gate_checks,
            "gate_passed": bool(all(gate_checks.values())),
            **timing,
        }
        parameters = {
            "min_rpm": 0.0,
            "hover_rpm": float(env.HOVER_RPM),
            "max_rpm": float(env.MAX_RPM),
            "kf": float(env.KF),
            "km": float(env.KM),
            "arm_length": float(env.L),
        }
        return episode, parameters
    finally:
        env.close()


def convert_arrays(arrays: dict[str, list]) -> dict[str, np.ndarray]:
    scalar_dtypes = {
        "episode_id": np.int32,
        "task_id": np.int8,
        "disturbance_profile_id": np.int8,
        "seed": np.int16,
        "step_id": np.int32,
        "sim_time": np.float64,
        "control_time": np.float64,
        "packet_version": np.int32,
        "packet_t_gen": np.float64,
        "packet_age": np.float32,
        "packet_age_ratio": np.float32,
        "packet_valid": np.bool_,
        "teacher_control_counter_before": np.int32,
        "teacher_control_counter_after": np.int32,
        "motor_saturation_mask": np.bool_,
    }
    return {
        name: np.asarray(values, dtype=scalar_dtypes.get(name, np.float32))
        for name, values in arrays.items()
    }


def validate_sample_axes(samples: dict[str, np.ndarray]) -> int:
    """Reject schema fields that are not aligned sample-for-sample."""

    if "context" not in samples or samples["context"].ndim != 2:
        raise ValueError("Stage2-v1 samples require a 2D context array.")
    sample_count = int(samples["context"].shape[0])
    mismatches = {
        name: int(values.shape[0]) if values.ndim else 0
        for name, values in samples.items()
        if values.ndim == 0 or int(values.shape[0]) != sample_count
    }
    if mismatches:
        raise AssertionError(
            f"Stage2-v1 sample-axis mismatch: expected={sample_count}, "
            f"got={mismatches}."
        )
    return sample_count


def _git_state(path: Path) -> dict:
    def run(*args):
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _collect_dataset(
    output_dir: Path,
    seeds: list[int],
    *,
    profile: str,
    dataset_version: str,
    disturbance_factory,
    teacher_mode: str,
    split_seed_groups: dict[str, list[int]],
    minimum_sample_count: int,
    candidate_for_training: bool,
) -> dict:
    output_dir = Path(output_dir)
    protected = [
        output_dir / "samples.npz",
        output_dir / "metadata.json",
        output_dir / "splits.npz",
        output_dir / "episode_manifest.jsonl",
    ]
    existing = [path for path in protected if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite Stage2-v1 data: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Stage2PilotConfig()
    arrays = _new_arrays()
    episodes = []
    physical_parameters = None
    seed_set = {int(seed) for seed in seeds}
    split_seed_sets = {
        name: {int(seed) for seed in values}
        for name, values in split_seed_groups.items()
    }
    flattened_split_seeds = [
        seed for values in split_seed_sets.values() for seed in values
    ]
    if set(flattened_split_seeds) != seed_set or len(flattened_split_seeds) != len(seed_set):
        raise ValueError("Split seed groups must be disjoint and cover every collection seed.")
    disturbances = {int(seed): disturbance_factory(int(seed)) for seed in seeds}
    disturbance_names = [disturbances[int(seed)].name for seed in seeds]
    disturbance_mapping = {
        name: index for index, name in enumerate(dict.fromkeys(disturbance_names))
    }
    episode_id = 0
    for seed in seeds:
        disturbance = disturbances[int(seed)]
        disturbance_id = disturbance_mapping[disturbance.name]
        for task in ("hover", "straight"):
            episode, parameters = collect_episode(
                episode_id=episode_id,
                task=task,
                seed=int(seed),
                disturbance_profile_id=disturbance_id,
                disturbance=disturbance,
                teacher_mode=teacher_mode,
                config=config,
                arrays=arrays,
            )
            episodes.append(episode)
            physical_parameters = parameters
            episode_id += 1
            print(
                f"[stage2-{profile}] completed {episode_id}/{2 * len(seeds)} "
                f"task={task} seed={seed} gate={episode['gate_passed']}",
                file=sys.stderr,
                flush=True,
            )

    samples = convert_arrays(arrays)
    validate_sample_axes(samples)
    if samples["context"].shape[1:] != (30,):
        raise AssertionError("Collected Stage2-v1 context is not 30D.")
    if not np.array_equal(samples["encoded_teacher_action"], samples["applied_action"]):
        raise AssertionError("Collected encoded and applied actions differ.")
    control_dt = 1.0 / config.control_frequency_hz
    if not np.allclose(samples["control_time"], samples["sim_time"] + control_dt):
        raise AssertionError("Collected control_time contract failed.")
    if not np.all(samples["packet_valid"]):
        raise AssertionError("Dataset contains invalid packets.")

    np.savez_compressed(output_dir / "samples.npz", **samples)
    episode_ids = samples["episode_id"]
    split_values = {}
    split_metadata = {}
    for split_name, split_seeds in split_seed_sets.items():
        split_episode_ids = np.asarray(
            [
                episode["episode_id"]
                for episode in episodes
                if int(episode["seed"]) in split_seeds
            ],
            dtype=np.int32,
        )
        split_sample_indices = np.flatnonzero(
            np.isin(episode_ids, split_episode_ids)
        ).astype(np.int64)
        split_values[f"{split_name}_episode_ids"] = split_episode_ids
        split_values[f"{split_name}_sample_indices"] = split_sample_indices
        split_metadata[split_name] = {
            "seeds": sorted(split_seeds),
            "episode_count": int(split_episode_ids.size),
            "sample_count": int(split_sample_indices.size),
        }
    np.savez(output_dir / "splits.npz", **split_values)
    (output_dir / "episode_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in episodes),
        encoding="utf-8",
    )
    sample_hash = hashlib.sha256((output_dir / "samples.npz").read_bytes()).hexdigest()
    project_state = _git_state(PROJECT_ROOT)
    environment_state = _git_state(PROJECT_ROOT.parent / "gym-pybullet-drones")
    canonical_teacher = teacher_mode == CANONICAL_WORLD_RATE_TEACHER
    max_hidden_contract_error = float(
        max(row["max_canonical_hidden_contract_error"] for row in episodes)
    )
    dataset_gate_checks = {
        "all_episode_gates": bool(all(row["gate_passed"] for row in episodes)),
        "minimum_sample_count": int(samples["context"].shape[0])
        >= int(minimum_sample_count),
        "context_reconstruction": float(
            max(row["max_context_reconstruction_error"] for row in episodes)
        )
        <= 1e-6,
        "rpm_roundtrip": float(
            max(row["max_rpm_roundtrip_error"] for row in episodes)
        )
        <= 0.01,
        "packets_valid": bool(np.all(samples["packet_valid"])),
        "motor_saturation_zero": not bool(np.any(samples["motor_saturation_mask"])),
        "split_seed_disjointness": len(flattened_split_seeds) == len(seed_set),
        "canonical_teacher_contract": (
            canonical_teacher and max_hidden_contract_error <= 1e-6
            if candidate_for_training
            else True
        ),
    }
    dataset_gate_passed = bool(all(dataset_gate_checks.values()))
    metadata = {
        "dataset_version": dataset_version,
        "profile": profile,
        "training_allowed": False,
        "training_release_requires": "quality_gate.json with ambiguity_gate_passed=true",
        "collection_gate_passed": dataset_gate_passed,
        "dataset_gate_passed": dataset_gate_passed,
        "dataset_gate_checks": dataset_gate_checks,
        "context_contract_id": STAGE2_V1_CONTEXT_ID,
        "context_dim": 30,
        "context_contract": stage2_v1_context_contract(),
        "reference_source": "external_packet",
        "reference_packet": "models.reference_packet.ReferencePacket",
        "packet_source": (
            "models.continuous_world_reference.ContinuousWorldPacketSource"
        ),
        "coordinate_frame": "world",
        "control_time_definition": "sim_time + 1/control_frequency_hz",
        "teacher_target": "current_reference_at_control_time",
        "teacher_contract_id": (
            CANONICAL_DSLPID_WORLD_RATE_ID if canonical_teacher else STATEFUL_TEACHER
        ),
        "future_reference_time": "control_time + 0.2 seconds",
        "motor_action_codec": ASYMMETRIC_RPM,
        "teacher_controller": (
            "models.canonical_dslpid_teacher.CanonicalDSLPIDTeacher"
            if canonical_teacher
            else "gym_pybullet_drones.control.DSLPIDControl.DSLPIDControl"
        ),
        "teacher_rate_mode": (
            WORLD_ANGULAR_VELOCITY if canonical_teacher else "episode_stateful"
        ),
        "teacher_hidden_state_policy": (
            "reset_integrals_each_label_and_infer_last_rpy_from_world_angular_velocity"
            if canonical_teacher
            else "episode_stateful"
        ),
        "teacher_reset_enabled": canonical_teacher,
        "teacher_action_noise": 0.0,
        "teacher_intervention_ratio": 1.0,
        "actor_control_ratio": 0.0,
        "config": asdict(config),
        "seeds": [int(seed) for seed in seeds],
        "task_id_mapping": TASK_IDS,
        "disturbance_profile_mapping": disturbance_mapping,
        "sample_count": int(samples["context"].shape[0]),
        "episode_count": len(episodes),
        "minimum_sample_count": int(minimum_sample_count),
        "splits": split_metadata,
        "episode_gate_passed": bool(all(row["gate_passed"] for row in episodes)),
        "pilot_gate_passed": bool(all(row["gate_passed"] for row in episodes)),
        "episode_gate_results": [
            {
                "episode_id": row["episode_id"],
                "task": row["task"],
                "seed": row["seed"],
                "gate_passed": row["gate_passed"],
            }
            for row in episodes
        ],
        "physical_parameters": physical_parameters,
        "max_context_reconstruction_error": float(
            max(row["max_context_reconstruction_error"] for row in episodes)
        ),
        "max_rpm_roundtrip_error": float(
            max(row["max_rpm_roundtrip_error"] for row in episodes)
        ),
        "max_canonical_hidden_contract_error": max_hidden_contract_error,
        "motor_saturation_ratio": float(np.mean(samples["motor_saturation_mask"])),
        "samples_sha256": sample_hash,
        "project_commit": project_state["commit"],
        "project_dirty": project_state["dirty"],
        "environment_commit": environment_state["commit"],
        "environment_dirty": environment_state["dirty"],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def collect_pilot(output_dir: Path, seeds: list[int]) -> dict:
    if sorted(int(seed) for seed in seeds) != [0, 1, 2]:
        raise ValueError("Stage2-v1 pilot requires exactly seeds 0 1 2.")
    return _collect_dataset(
        output_dir,
        [int(seed) for seed in seeds],
        profile="pilot_diagnostic_only",
        dataset_version=PILOT_DATASET_VERSION,
        disturbance_factory=disturbance_for_seed,
        teacher_mode=STATEFUL_TEACHER,
        split_seed_groups={f"pilot_seed_{seed}": [int(seed)] for seed in seeds},
        minimum_sample_count=1,
        candidate_for_training=False,
    )


def collect_formal(output_dir: Path, seeds: list[int]) -> dict:
    normalized_seeds = [int(seed) for seed in seeds]
    if normalized_seeds != list(FORMAL_SEEDS):
        raise ValueError(
            "Formal Stage2-v1 collection requires ordered seeds 0..50 exactly."
        )
    split_seed_groups = {
        split_name: [
            seed for seed in normalized_seeds if formal_split_for_seed(seed) == split_name
        ]
        for split_name in ("train", "validation", "test")
    }
    return _collect_dataset(
        output_dir,
        normalized_seeds,
        profile="formal_bc_candidate",
        dataset_version=FORMAL_DATASET_VERSION,
        disturbance_factory=formal_disturbance_for_seed,
        teacher_mode=CANONICAL_WORLD_RATE_TEACHER,
        split_seed_groups=split_seed_groups,
        minimum_sample_count=250_000,
        candidate_for_training=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("pilot", "formal"),
        default="formal",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    if args.profile == "pilot":
        output_dir = Path(
            args.output_dir or "data/behavior_cloning/stage2_v1_external_packet_pilot_v2"
        )
        metadata = collect_pilot(output_dir, args.seeds or [0, 1, 2])
    else:
        output_dir = Path(
            args.output_dir
            or "data/behavior_cloning/stage2_v1_external_packet_canonical_v2"
        )
        metadata = collect_formal(output_dir, args.seeds or list(FORMAL_SEEDS))
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
