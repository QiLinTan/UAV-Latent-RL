"""Stage 1 gate: continuous world trajectory through ReferencePacket and DSLPID.

This evaluator is intentionally independent of every RL agent and planner.  A
single deterministic global trajectory is created per episode.  The 15 Hz
producer publishes sliding one-second windows from that trajectory while the
120 Hz lower loop interpolates the newest packet with simulation time.
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
import pybullet as p


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.gym_pybullet_compat import ensure_gym_pybullet_envs_compat

ensure_gym_pybullet_envs_compat()

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from envs.ForestAviary import CustomForestAviary
from models.continuous_world_reference import (
    ContinuousWorldPacketSource,
    TrapezoidalWorldTrajectory,
    derive_motion_timing,
)
from models.motor_action_codec import (
    ASYMMETRIC_RPM,
    MotorActionCodec,
    MotorPhysicalLimits,
)
from models.reference_packet import AsyncReferenceBuffer


@dataclass(frozen=True)
class Stage1Config:
    start_position: tuple[float, float, float] = (-3.5, 0.0, 1.0)
    target_position: tuple[float, float, float] = (3.5, 0.0, 1.0)
    max_speed: float = 0.6
    max_acceleration: float = 1.5
    high_frequency_hz: int = 15
    low_frequency_hz: int = 120
    window_seconds: float = 1.0
    sequence_length: int = 16
    settle_seconds: float = 2.0
    goal_tolerance: float = 0.20
    attitude_limit: float = 0.65

    def __post_init__(self):
        if min(self.max_speed, self.max_acceleration, self.window_seconds) <= 0.0:
            raise ValueError("Trajectory speed, acceleration, and window must be positive.")
        if self.high_frequency_hz <= 0 or self.low_frequency_hz <= 0:
            raise ValueError("Control frequencies must be positive.")
        if self.low_frequency_hz % self.high_frequency_hz != 0:
            raise ValueError("Low frequency must be divisible by high frequency.")
        if self.sequence_length < 2:
            raise ValueError("Reference windows require at least two samples.")
        if self.settle_seconds <= 0.0:
            raise ValueError("Settle time must be positive.")


def make_motor_codec(env) -> MotorActionCodec:
    codec = MotorActionCodec(
        MotorPhysicalLimits(
            min_rpm=0.0,
            hover_rpm=float(env.HOVER_RPM),
            max_rpm=float(env.MAX_RPM),
            kf=float(env.KF),
        ),
        mode=ASYMMETRIC_RPM,
    )
    env.set_motor_action_codec(codec)
    return codec


def derive_timing(config: Stage1Config, trajectory: TrapezoidalWorldTrajectory) -> dict:
    return derive_motion_timing(
        trajectory,
        max_speed=config.max_speed,
        settle_seconds=config.settle_seconds,
    )


def _quiet_env(**kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return CustomForestAviary(**kwargs)


def run_episode(seed: int, config: Stage1Config) -> dict:
    trajectory = TrapezoidalWorldTrajectory(
        config.start_position,
        config.target_position,
        max_speed=config.max_speed,
        max_acceleration=config.max_acceleration,
    )
    timing = derive_timing(config, trajectory)
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
        episode_len_sec=timing["timeout"],
        goal_tolerance=config.goal_tolerance,
    )
    try:
        env.reset(seed=int(seed))
        controller = DSLPIDControl(drone_model=env.DRONE_MODEL)
        controller.reset()
        codec = make_motor_codec(env)
        buffer = AsyncReferenceBuffer()
        source = ContinuousWorldPacketSource(
            trajectory,
            window_seconds=config.window_seconds,
            sequence_length=config.sequence_length,
        )

        low_dt = 1.0 / config.low_frequency_hz
        high_interval = config.low_frequency_hz // config.high_frequency_hz
        position_squared_errors: list[float] = []
        velocity_squared_errors: list[float] = []
        height_abs_errors: list[float] = []
        packet_ages: list[float] = []
        accepted_versions: list[int] = []
        max_roll = 0.0
        max_pitch = 0.0
        attitude_violation_count = 0
        saturated_motor_values = 0
        motor_value_count = 0
        missing_packet_count = 0
        stale_packet_count = 0
        fallback_count = 0
        step_count = 0
        info = {}
        terminated = truncated = False
        last_target_position = np.asarray(config.start_position, dtype=np.float32)

        while not (terminated or truncated):
            now = step_count * low_dt
            if step_count % high_interval == 0:
                packet = source.make_packet(now)
                if buffer.publish(packet):
                    accepted_versions.append(int(packet.version))

            control_time = now + low_dt
            if not buffer.has_value:
                missing_packet_count += 1
                fallback_count += 1
                target_position = last_target_position
                target_velocity = np.zeros(3, dtype=np.float32)
                sample = None
            else:
                sample = buffer.sample(control_time, lookahead_seconds=0.0)
                if not sample["valid"]:
                    stale_packet_count += 1
                    fallback_count += 1
                    target_position = last_target_position
                    target_velocity = np.zeros(3, dtype=np.float32)
                else:
                    target_position = np.asarray(
                        sample["current_position"], dtype=np.float32
                    )
                    target_velocity = np.asarray(
                        sample["current_velocity"], dtype=np.float32
                    )
                    last_target_position = target_position.copy()
                    packet_ages.append(float(sample["age_seconds"]))

            rpm, _, _ = controller.computeControl(
                control_timestep=low_dt,
                cur_pos=np.asarray(env.pos[0], dtype=np.float64),
                cur_quat=np.asarray(
                    p.getQuaternionFromEuler(env.rpy[0]), dtype=np.float64
                ),
                cur_vel=np.asarray(env.vel[0], dtype=np.float64),
                cur_ang_vel=np.asarray(env.ang_v[0], dtype=np.float64),
                target_pos=np.asarray(target_position, dtype=np.float64),
                target_vel=np.asarray(target_velocity, dtype=np.float64),
            )
            action = codec.rpm_to_normalized_action(rpm).astype(np.float32)
            saturated_motor_values += int(np.sum(np.abs(action) >= 1.0 - 1e-6))
            motor_value_count += int(action.size)

            _, _, terminated, truncated, info = env.step(action.reshape(1, -1))
            actual_position = np.asarray(env.pos[0], dtype=np.float64)
            actual_velocity = np.asarray(env.vel[0], dtype=np.float64)
            position_squared_errors.append(
                float(np.sum((actual_position - target_position) ** 2))
            )
            velocity_squared_errors.append(
                float(np.sum((actual_velocity - target_velocity) ** 2))
            )
            height_abs_errors.append(
                abs(float(actual_position[2] - target_position[2]))
            )
            roll = abs(float(env.rpy[0][0]))
            pitch = abs(float(env.rpy[0][1]))
            max_roll = max(max_roll, roll)
            max_pitch = max(max_pitch, pitch)
            attitude_violation_count += int(
                roll > config.attitude_limit or pitch > config.attitude_limit
            )
            step_count += 1

        expected_versions = list(range(1, len(accepted_versions) + 1))
        versions_continuous = accepted_versions == expected_versions
        final_distance = float(
            np.linalg.norm(
                np.asarray(config.target_position, dtype=np.float64)
                - np.asarray(env.pos[0], dtype=np.float64)
            )
        )
        packet_age_values = np.asarray(packet_ages, dtype=np.float64)
        height_values = np.asarray(height_abs_errors, dtype=np.float64)
        return {
            "seed": int(seed),
            "success": bool(info.get("success", False)),
            "done_reason": str(info.get("done_reason", "unknown")),
            "steps": int(step_count),
            "final_distance": final_distance,
            "position_rmse": float(np.sqrt(np.mean(position_squared_errors))),
            "velocity_tracking_error": float(
                np.sqrt(np.mean(velocity_squared_errors))
            ),
            "max_roll": float(max_roll),
            "max_pitch": float(max_pitch),
            "attitude_violation_count": int(attitude_violation_count),
            "height_abs_error": float(height_values.mean()),
            "height_rmse": float(np.sqrt(np.mean(np.square(height_values)))),
            "motor_saturation_ratio": float(
                saturated_motor_values / max(1, motor_value_count)
            ),
            "packet_age_mean": float(packet_age_values.mean()),
            "packet_age_p95": float(np.percentile(packet_age_values, 95.0)),
            "packet_age_max": float(packet_age_values.max()),
            "missing_packet_count": int(missing_packet_count),
            "stale_packet_count": int(stale_packet_count),
            "fallback_count": int(fallback_count),
            "packet_version_continuity": bool(versions_continuous),
            "accepted_packet_count": int(len(accepted_versions)),
            "last_packet_version": int(accepted_versions[-1]),
            **timing,
        }
    finally:
        env.close()


GATE_THRESHOLDS = {
    "success_rate": 1.0,
    "max_final_distance": 0.20,
    "max_position_rmse": 0.10,
    "max_velocity_tracking_error": 0.15,
    "max_attitude_radians": 0.65,
    "max_attitude_violation_count": 0,
    "max_height_rmse": 0.05,
    "max_motor_saturation_ratio": 0.05,
    "max_fallback_count": 0,
}


def summarize(episodes: list[dict], config: Stage1Config) -> dict:
    if not episodes:
        raise ValueError("At least one Stage 1 episode is required.")

    def mean(key: str) -> float:
        return float(np.mean([episode[key] for episode in episodes]))

    summary = {
        "episode_count": len(episodes),
        "success_rate": mean("success"),
        "final_distance": mean("final_distance"),
        "position_rmse": mean("position_rmse"),
        "velocity_tracking_error": mean("velocity_tracking_error"),
        "max_roll": float(max(episode["max_roll"] for episode in episodes)),
        "max_pitch": float(max(episode["max_pitch"] for episode in episodes)),
        "attitude_violation_count": int(
            sum(episode["attitude_violation_count"] for episode in episodes)
        ),
        "height_abs_error": mean("height_abs_error"),
        "height_rmse": mean("height_rmse"),
        "motor_saturation_ratio": mean("motor_saturation_ratio"),
        "packet_age_mean": mean("packet_age_mean"),
        "packet_age_p95": mean("packet_age_p95"),
        "packet_age_max": float(max(episode["packet_age_max"] for episode in episodes)),
        "missing_packet_count": int(
            sum(episode["missing_packet_count"] for episode in episodes)
        ),
        "stale_packet_count": int(
            sum(episode["stale_packet_count"] for episode in episodes)
        ),
        "fallback_count": int(sum(episode["fallback_count"] for episode in episodes)),
        "packet_version_continuity": bool(
            all(episode["packet_version_continuity"] for episode in episodes)
        ),
        "trajectory_duration": episodes[0]["trajectory_duration"],
        "timeout": episodes[0]["timeout"],
        "reachability_assertion": bool(
            all(episode["reachability_assertion"] for episode in episodes)
        ),
    }
    checks = {
        "success_rate": summary["success_rate"] >= GATE_THRESHOLDS["success_rate"],
        "final_distance": summary["final_distance"] <= GATE_THRESHOLDS["max_final_distance"],
        "position_rmse": summary["position_rmse"] <= GATE_THRESHOLDS["max_position_rmse"],
        "velocity_tracking_error": (
            summary["velocity_tracking_error"]
            <= GATE_THRESHOLDS["max_velocity_tracking_error"]
        ),
        "attitude": (
            summary["max_roll"] <= GATE_THRESHOLDS["max_attitude_radians"]
            and summary["max_pitch"] <= GATE_THRESHOLDS["max_attitude_radians"]
            and summary["attitude_violation_count"]
            <= GATE_THRESHOLDS["max_attitude_violation_count"]
        ),
        "height": summary["height_rmse"] <= GATE_THRESHOLDS["max_height_rmse"],
        "motor_saturation": (
            summary["motor_saturation_ratio"]
            <= GATE_THRESHOLDS["max_motor_saturation_ratio"]
        ),
        "packets": (
            summary["missing_packet_count"] == 0
            and summary["stale_packet_count"] == 0
            and summary["fallback_count"] <= GATE_THRESHOLDS["max_fallback_count"]
            and summary["packet_version_continuity"]
        ),
        "reachability": summary["reachability_assertion"],
    }
    summary["gate_checks"] = checks
    summary["stage1_straight_line_gate_passed"] = bool(all(checks.values()))
    summary["failure_classification"] = classify_failures(summary, checks, config)
    return summary


def classify_failures(summary: dict, checks: dict, config: Stage1Config) -> list[str]:
    failures: list[str] = []
    if not checks["reachability"]:
        failures.append("reference generation error")
    if not checks["packets"]:
        failures.append("packet/interpolation error")
    if not checks["motor_saturation"]:
        failures.append("motor saturation")
    if not checks["position_rmse"] or not checks["velocity_tracking_error"] or not checks["attitude"] or not checks["height"]:
        failures.append("PID tracking limitation")
    inconsistent_success = any(
        episode["success"] != (episode["final_distance"] <= config.goal_tolerance)
        for episode in summary.get("episodes", [])
    )
    if inconsistent_success or (not checks["success_rate"] and checks["position_rmse"]):
        failures.append("success/termination definition error")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--output",
        default="analysis_outputs/stage1_reference_pid/straight_line_results.json",
    )
    args = parser.parse_args()
    if len(args.seeds) < 3:
        raise ValueError("The formal Stage 1 gate requires at least three fixed seeds.")

    config = Stage1Config()
    episodes = [run_episode(seed, config) for seed in args.seeds]
    summary = summarize(episodes, config)
    # Classification can inspect per-episode success consistency without adding
    # the bulky episode list to its public summary input contract.
    summary["failure_classification"] = classify_failures(
        {**summary, "episodes": episodes}, summary["gate_checks"], config
    )
    result = {
        "contract": "stage1_continuous_world_reference_v1",
        "controller": "DSLPIDControl",
        "reference_packet": "models.reference_packet.ReferencePacket",
        "reference_buffer": "models.reference_packet.AsyncReferenceBuffer",
        "async_reference_cache_used": False,
        "rl_or_planner_used": False,
        "config": asdict(config),
        "gate_thresholds": GATE_THRESHOLDS,
        "episodes": episodes,
        "summary": summary,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
