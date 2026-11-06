"""Compare stateful and canonical DSLPID teachers on Stage2-v1 packets."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.gym_pybullet_compat import ensure_gym_pybullet_envs_compat

ensure_gym_pybullet_envs_compat()

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from models.canonical_dslpid_teacher import (
    CanonicalDSLPIDTeacher,
    WORLD_ANGULAR_VELOCITY,
    ZERO_RATE,
)
from models.continuous_world_reference import ContinuousWorldPacketSource
from scripts.collect_stage2_v1_dataset import (
    Stage2PilotConfig,
    apply_initial_disturbance,
    disturbance_for_seed,
    make_agent,
    make_env,
    make_task_trajectory,
)
from envs.preprocess import preprocess_state


STATEFUL = "stateful"
CANONICAL_ZERO_RATE = "canonical_zero_rate"
CANONICAL_WORLD_ANGULAR_VELOCITY = "canonical_world_angular_velocity"
TEACHER_MODES = (
    STATEFUL,
    CANONICAL_ZERO_RATE,
    CANONICAL_WORLD_ANGULAR_VELOCITY,
)


def make_teacher(mode: str, drone_model):
    if mode == STATEFUL:
        teacher = DSLPIDControl(drone_model=drone_model)
        teacher.reset()
        return teacher
    if mode == CANONICAL_ZERO_RATE:
        return CanonicalDSLPIDTeacher(drone_model, rate_mode=ZERO_RATE)
    if mode == CANONICAL_WORLD_ANGULAR_VELOCITY:
        return CanonicalDSLPIDTeacher(
            drone_model,
            rate_mode=WORLD_ANGULAR_VELOCITY,
        )
    raise ValueError(f"Unknown teacher mode {mode!r}.")


def run_episode(
    *,
    mode: str,
    task: str,
    seed: int,
    config: Stage2PilotConfig,
) -> dict:
    trajectory, timeout, timing = make_task_trajectory(task, config)
    disturbance = disturbance_for_seed(seed)
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
        teacher = make_teacher(mode, env.DRONE_MODEL)
        source = ContinuousWorldPacketSource(
            trajectory,
            window_seconds=config.window_seconds,
            sequence_length=config.sequence_length,
        )

        low_dt = 1.0 / config.control_frequency_hz
        step = 0
        terminated = truncated = False
        info = {}
        position_squared_errors = []
        velocity_squared_errors = []
        height_squared_errors = []
        accepted_versions = []
        action_delta_squared = []
        previous_action = np.zeros(4, dtype=np.float32)
        saturation_count = 0
        motor_value_count = 0
        max_roll = max_pitch = 0.0
        attitude_violation_count = 0
        missing_packet_count = stale_packet_count = fallback_count = 0

        while not (terminated or truncated):
            sim_time = step * low_dt
            if step % config.packet_interval_steps == 0:
                packet = source.make_packet(sim_time)
                if not agent.publish_reference(packet):
                    raise RuntimeError("Teacher probe packet publication was rejected.")
                accepted_versions.append(int(packet.version))

            context = agent.prepare_runtime_context(state)
            if context.shape != (30,):
                raise AssertionError("Teacher probe did not use the frozen 30D context.")
            reference = agent.last_reference_sample
            if reference is None:
                missing_packet_count += 1
                raise RuntimeError("Teacher probe encountered a missing packet.")
            if not bool(reference["valid"]):
                stale_packet_count += 1
                raise RuntimeError("Teacher probe encountered a stale packet.")
            fallback_count += int(
                bool(agent.last_runtime_info.get("async_degraded_fallback", 0.0))
            )

            drone_state = env._getDroneStateVector(0)
            rpm, _, _ = teacher.computeControl(
                control_timestep=env.CTRL_TIMESTEP,
                cur_pos=drone_state[0:3],
                cur_quat=drone_state[3:7],
                cur_vel=drone_state[10:13],
                cur_ang_vel=drone_state[13:16],
                target_pos=reference["current_position"],
                target_vel=reference["current_velocity"],
            )
            action = codec.rpm_to_normalized_action(rpm).astype(np.float32)
            saturation_count += int(np.sum(np.abs(action) >= 1.0 - 1e-6))
            motor_value_count += 4
            action_delta_squared.append(float(np.mean(np.square(action - previous_action))))
            previous_action = action.copy()

            next_observation, _, terminated, truncated, info = env.step(
                action.reshape(1, -1)
            )
            actual_position = np.asarray(env.pos[0], dtype=np.float64)
            actual_velocity = np.asarray(env.vel[0], dtype=np.float64)
            reference_position = np.asarray(reference["current_position"], dtype=np.float64)
            reference_velocity = np.asarray(reference["current_velocity"], dtype=np.float64)
            position_squared_errors.append(
                float(np.sum(np.square(actual_position - reference_position)))
            )
            velocity_squared_errors.append(
                float(np.sum(np.square(actual_velocity - reference_velocity)))
            )
            height_squared_errors.append(
                float(np.square(actual_position[2] - reference_position[2]))
            )
            roll = abs(float(env.rpy[0][0]))
            pitch = abs(float(env.rpy[0][1]))
            max_roll = max(max_roll, roll)
            max_pitch = max(max_pitch, pitch)
            attitude_violation_count += int(
                roll > config.attitude_limit or pitch > config.attitude_limit
            )

            agent.record_executed_action(action)
            agent.advance_runtime_step()
            state = preprocess_state(next_observation.reshape(-1))
            step += 1

        position_rmse = float(np.sqrt(np.mean(position_squared_errors)))
        velocity_error = float(np.sqrt(np.mean(velocity_squared_errors)))
        height_rmse = float(np.sqrt(np.mean(height_squared_errors)))
        final_distance = float(
            np.linalg.norm(
                np.asarray(config.target_position, dtype=np.float64)
                - np.asarray(env.pos[0], dtype=np.float64)
            )
        )
        done_reason = str(info.get("done_reason", "unknown"))
        success = (
            done_reason == "timeout"
            if task == "hover"
            else bool(info.get("success", False))
        )
        position_limit = 0.05 if task == "hover" else 0.10
        height_limit = 0.03 if task == "hover" else 0.05
        versions_continuous = accepted_versions == list(
            range(1, len(accepted_versions) + 1)
        )
        checks = {
            "success": bool(success),
            "final_distance": task == "hover" or final_distance <= 0.20,
            "position_rmse": position_rmse <= position_limit,
            "velocity_tracking_error": velocity_error <= 0.15,
            "attitude": max_roll <= config.attitude_limit
            and max_pitch <= config.attitude_limit
            and attitude_violation_count == 0,
            "height": height_rmse <= height_limit,
            "motor_saturation": saturation_count / max(1, motor_value_count) <= 0.05,
            "packets": missing_packet_count == 0
            and stale_packet_count == 0
            and fallback_count == 0
            and versions_continuous,
        }
        return {
            "mode": mode,
            "task": task,
            "seed": int(seed),
            "disturbance": disturbance.name,
            "success": bool(success),
            "done_reason": done_reason,
            "steps": step,
            "final_distance": final_distance,
            "position_rmse": position_rmse,
            "velocity_tracking_error": velocity_error,
            "height_rmse": height_rmse,
            "max_roll": max_roll,
            "max_pitch": max_pitch,
            "attitude_violation_count": attitude_violation_count,
            "action_delta_rms": float(np.sqrt(np.mean(action_delta_squared))),
            "motor_saturation_ratio": saturation_count / max(1, motor_value_count),
            "missing_packet_count": missing_packet_count,
            "stale_packet_count": stale_packet_count,
            "fallback_count": fallback_count,
            "packet_version_continuity": versions_continuous,
            "gate_checks": checks,
            "gate_passed": bool(all(checks.values())),
            **timing,
        }
    finally:
        env.close()


def summarize_mode(mode: str, episodes: list[dict]) -> dict:
    checks = {
        "all_six_episodes_pass": all(row["gate_passed"] for row in episodes),
        "hover_pass": all(
            row["gate_passed"] for row in episodes if row["task"] == "hover"
        ),
        "straight_pass": all(
            row["gate_passed"] for row in episodes if row["task"] == "straight"
        ),
        "no_saturation": all(row["motor_saturation_ratio"] == 0.0 for row in episodes),
        "packets_clean": all(
            row["missing_packet_count"] == 0
            and row["stale_packet_count"] == 0
            and row["fallback_count"] == 0
            and row["packet_version_continuity"]
            for row in episodes
        ),
    }
    return {
        "mode": mode,
        "episode_count": len(episodes),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def run_probe(seeds: list[int]) -> dict:
    if sorted(int(seed) for seed in seeds) != [0, 1, 2]:
        raise ValueError("Teacher determinization probe requires seeds 0 1 2.")
    config = Stage2PilotConfig()
    episodes = []
    for mode in TEACHER_MODES:
        for seed in seeds:
            for task in ("hover", "straight"):
                row = run_episode(
                    mode=mode,
                    task=task,
                    seed=int(seed),
                    config=config,
                )
                episodes.append(row)
                print(
                    f"[teacher-probe] mode={mode} task={task} seed={seed} "
                    f"gate={row['gate_passed']}",
                    file=sys.stderr,
                    flush=True,
                )
    summaries = {
        mode: summarize_mode(
            mode,
            [row for row in episodes if row["mode"] == mode],
        )
        for mode in TEACHER_MODES
    }
    viable = [
        mode
        for mode in (CANONICAL_WORLD_ANGULAR_VELOCITY, CANONICAL_ZERO_RATE)
        if summaries[mode]["passed"]
    ]
    return {
        "contract": "stage2_v1_teacher_determinization_probe",
        "config": asdict(config),
        "reference_source": "external_packet",
        "teacher_target": "current_reference_at_control_time",
        "network_training_performed": False,
        "episodes": episodes,
        "mode_summaries": summaries,
        "viable_canonical_modes": viable,
        "teacher_determinization_gate_passed": bool(viable),
        "recommended_teacher_mode": viable[0] if viable else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--output",
        default=(
            "analysis_outputs/stage2_v1_teacher_determinization/"
            "probe_results.json"
        ),
    )
    args = parser.parse_args()
    result = run_probe(args.seeds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}.")
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
