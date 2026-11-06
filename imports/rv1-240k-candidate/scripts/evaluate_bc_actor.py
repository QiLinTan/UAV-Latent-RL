from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

from utils.gym_pybullet_compat import ensure_gym_pybullet_envs_compat

ensure_gym_pybullet_envs_compat()

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from algos.td3.td3_reference_tracking import TD3ReferenceTracking
from data.bc_metrics import decode_asymmetric_rpm, rpm_to_wrench
from envs.preprocess import preprocess_state
from models.motor_action_codec import ASYMMETRIC_RPM
from scripts.bc_scenarios import (
    BCScenario,
    apply_initial_disturbance,
    apply_runtime_impulse,
    build_scenarios,
    impulse_step,
    make_env,
    make_reference_packet,
    make_reference_window,
    recovery_state,
)


UNSTABLE_REASONS = {"attitude_bound", "height_bound", "collision", "xy_bound"}


def make_agent(env, checkpoint, device):
    agent = TD3ReferenceTracking(
        state_dim=int(np.prod(env.observation_space.shape)),
        action_dim=4,
        max_action=1.0,
        target_position=[3.5, 0.0, 1.0],
        ctrl_freq=120,
        sequence_length=15,
        reference_horizon_seconds=1.0,
        high_level_interval=8,
        reference_mode="hover",
        max_reference_speed=0.6,
        max_reference_acceleration=1.5,
        max_reference_vertical_speed=0.4,
        actor_structure="plain",
        lower_action_history_steps=0,
        motor_action_mode=ASYMMETRIC_RPM,
        device=device,
    )
    if int(checkpoint["architecture"]["input_dim"]) != agent.context_dim:
        raise ValueError(
            f"Checkpoint expects context {checkpoint['architecture']['input_dim']}, "
            f"evaluation agent has {agent.context_dim}."
        )
    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.actor.eval()
    return agent


def scenario_lookup():
    return {scenario.name: scenario for scenario in build_scenarios()}


def development_episodes():
    scenarios = scenario_lookup()
    episodes = [
        (scenarios["nominal_hover"], seed, 12.0)
        for seed in range(10, 15)
    ]
    for name in (
        "initial_roll_-0.05",
        "initial_roll_+0.05",
        "initial_pitch_-0.05",
        "initial_pitch_+0.05",
        "initial_yaw_-0.05",
        "initial_yaw_+0.05",
    ):
        episodes.extend((scenarios[name], seed, 12.0) for seed in range(10, 15))
    return episodes


def formal_episodes():
    scenarios = scenario_lookup()
    episodes = [
        (scenarios["nominal_hover"], seed, 30.0)
        for seed in range(20, 25)
    ]
    selected = [
        ("initial_roll_+0.10", 20),
        ("initial_roll_-0.10", 21),
        ("initial_pitch_+0.10", 22),
        ("initial_pitch_-0.10", 23),
        ("initial_yaw_+0.10", 24),
        ("initial_mixed_rpy_+0.10", 25),
        ("initial_mixed_rpy_-0.10", 26),
        ("holdout_mixed_rpy", 27),
        ("initial_velocity_x_pos", 28),
        ("initial_velocity_y_neg", 29),
        ("initial_angular_velocity_random", 30),
        ("holdout_diagonal_velocity", 31),
        ("initial_height_+0.20", 32),
        ("initial_height_-0.20", 33),
        ("impulse_linear_mixed", 34),
        ("impulse_angular_mixed", 35),
        ("impulse_combined", 36),
        ("holdout_combined_impulse", 37),
        ("nominal_height_step_up", 38),
        ("nominal_height_step_down", 39),
        ("nominal_low_speed_line", 40),
        ("nominal_gentle_curve", 41),
        ("holdout_reverse_arc", 42),
    ]
    episodes.extend((scenarios[name], seed, 12.0) for name, seed in selected)
    return episodes


def first_deviation_channel(
    actor_action,
    teacher_action,
    actor_wrench,
    teacher_wrench,
    *,
    hover_total_thrust,
    torque_thresholds,
):
    candidates = []
    action_error = np.abs(actor_action - teacher_action)
    motor_index = int(np.argmax(action_error))
    if action_error[motor_index] >= 0.08:
        candidates.append(
            (
                float(action_error[motor_index] / 0.08),
                f"motor_{motor_index}_action",
                float(action_error[motor_index]),
            )
        )
    collective_error = abs(float(actor_wrench[0] - teacher_wrench[0]))
    if collective_error >= 0.05 * hover_total_thrust:
        candidates.append(
            (
                collective_error / (0.05 * hover_total_thrust),
                "collective_thrust",
                collective_error,
            )
        )
    for index, name in enumerate(("roll", "pitch", "yaw"), start=1):
        threshold = float(torque_thresholds[name])
        teacher_value = float(teacher_wrench[index])
        actor_value = float(actor_wrench[index])
        if abs(teacher_value) >= threshold and np.sign(teacher_value) != np.sign(
            actor_value
        ):
            candidates.append((2.0, f"{name}_torque_direction", actor_value - teacher_value))
        torque_error = abs(actor_value - teacher_value)
        if torque_error >= max(1e-10, 2.0 * threshold):
            candidates.append(
                (
                    torque_error / max(1e-10, 2.0 * threshold),
                    f"{name}_torque_magnitude",
                    torque_error,
                )
            )
    if not candidates:
        return None
    _, channel, magnitude = max(candidates, key=lambda item: item[0])
    return {"channel": channel, "magnitude": float(magnitude)}


def run_episode(
    scenario: BCScenario,
    seed: int,
    duration: float,
    *,
    controller_mode: str,
    checkpoint,
    device,
    physical_parameters,
    torque_thresholds,
    critical_torque_thresholds=None,
    episode_uid: int,
):
    if critical_torque_thresholds is None:
        critical_torque_thresholds = torque_thresholds
    env = make_env(scenario, duration)
    obs, _ = env.reset(seed=seed)
    apply_initial_disturbance(env, scenario, seed)
    obs = env._computeObs()
    state = preprocess_state(obs.reshape(-1))
    agent = make_agent(env, checkpoint, device)
    agent.reset_episode()
    codec = agent.configure_motor_action_interface(env)
    agent.set_high_level_enabled(False)
    teacher = DSLPIDControl(drone_model=env.DRONE_MODEL)
    teacher.reset()
    full_reference = make_reference_packet(scenario.reference_kind, duration)
    event_step = impulse_step(scenario, env.CTRL_FREQ)
    recovery_active = scenario.category == "initial_recovery"
    recovery_start = 0 if recovery_active else None
    stable_count = 0
    recovery_time = None
    reference_version = 0
    steps = 0
    info = {}
    max_rpy = np.zeros(3)
    max_angular_velocity = 0.0
    max_position_error = 0.0
    final_position_error = 0.0
    position_errors = []
    action_errors = []
    wrench_errors = []
    torque_direction_matches = [[], [], []]
    torque_direction_eligible_counts = [0, 0, 0]
    first_deviation = None
    step_rows = {
        "episode_uid": [],
        "step_id": [],
        "time": [],
        "context": [],
        "position_error": [],
        "rpy": [],
        "angular_velocity": [],
        "actor_action": [],
        "teacher_action": [],
        "applied_action": [],
        "actor_wrench": [],
        "teacher_wrench": [],
        "recovery_active": [],
        "recovery_early": [],
    }
    hover_total_thrust = (
        4.0
        * float(physical_parameters["kf"])
        * float(physical_parameters["hover_rpm"]) ** 2
    )

    while True:
        if event_step is not None and steps == event_step:
            apply_runtime_impulse(env, scenario, seed)
            obs = env._computeObs()
            state = preprocess_state(obs.reshape(-1))
            recovery_active = True
            recovery_start = steps
            stable_count = 0

        if steps % agent.high_level_interval == 0:
            reference_version += 1
            window = make_reference_window(
                full_reference,
                now=agent.runtime_step * agent.control_dt,
                version=reference_version,
                horizon_seconds=agent.reference_horizon_seconds,
                sequence_length=agent.sequence_length,
            )
            if not agent.reference_buffer.publish(window):
                raise RuntimeError("Reference window publication was rejected.")

        context = agent.prepare_runtime_context(state)
        reference_sample = agent.last_reference_sample
        if reference_sample is None or not reference_sample["valid"]:
            raise RuntimeError("Actor evaluation encountered an invalid reference.")
        recovery_active_at_action = recovery_active
        recovery_early_at_action = bool(
            recovery_active
            and recovery_start is not None
            and steps - int(recovery_start) < int(env.CTRL_FREQ)
        )
        if recovery_active:
            stable_now = recovery_state(env, reference_sample)
            stable_count = stable_count + 1 if stable_now else 0
            if stable_count >= 30:
                first_stable_step = steps - stable_count + 1
                recovery_time = (
                    first_stable_step - int(recovery_start)
                ) / float(env.CTRL_FREQ)
                recovery_active = False

        drone_state = env._getDroneStateVector(0)
        raw_teacher_rpm, _, _ = teacher.computeControl(
            control_timestep=env.CTRL_TIMESTEP,
            cur_pos=drone_state[0:3],
            cur_quat=drone_state[3:7],
            cur_vel=drone_state[10:13],
            cur_ang_vel=drone_state[13:16],
            target_pos=reference_sample["lookahead_position"],
            target_vel=reference_sample["lookahead_velocity"],
        )
        teacher_action = codec.rpm_to_normalized_action(raw_teacher_rpm).astype(
            np.float32
        )
        with torch.no_grad():
            actor_action = agent.actor(
                torch.as_tensor(
                    context,
                    dtype=torch.float32,
                    device=device,
                ).reshape(1, -1)
            ).cpu().numpy().reshape(-1)
        actor_action = agent.constrain_motor_action(actor_action)
        applied_action = (
            teacher_action.copy()
            if controller_mode == "teacher"
            else actor_action.copy()
        )

        actor_rpm = decode_asymmetric_rpm(
            actor_action.reshape(1, -1),
            physical_parameters,
        )[0]
        teacher_rpm = decode_asymmetric_rpm(
            teacher_action.reshape(1, -1),
            physical_parameters,
        )[0]
        actor_wrench = rpm_to_wrench(
            actor_rpm.reshape(1, -1),
            physical_parameters,
        )[0]
        teacher_wrench = rpm_to_wrench(
            teacher_rpm.reshape(1, -1),
            physical_parameters,
        )[0]
        if controller_mode == "actor" and first_deviation is None:
            deviation = first_deviation_channel(
                actor_action,
                teacher_action,
                actor_wrench,
                teacher_wrench,
                hover_total_thrust=hover_total_thrust,
                torque_thresholds=critical_torque_thresholds,
            )
            if deviation is not None:
                first_deviation = {
                    "time": steps / float(env.CTRL_FREQ),
                    **deviation,
                }

        current_position_error = float(
            np.linalg.norm(
                drone_state[0:3] - reference_sample["current_position"]
            )
        )
        position_errors.append(current_position_error)
        action_errors.append(actor_action - teacher_action)
        wrench_errors.append(actor_wrench - teacher_wrench)
        for torque_index, name in enumerate(("roll", "pitch", "yaw"), start=1):
            if recovery_active_at_action and abs(
                float(teacher_wrench[torque_index])
            ) >= float(critical_torque_thresholds[name]):
                torque_direction_eligible_counts[torque_index - 1] += 1
                torque_direction_matches[torque_index - 1].append(
                    np.sign(teacher_wrench[torque_index])
                    == np.sign(actor_wrench[torque_index])
                )
        step_rows["episode_uid"].append(episode_uid)
        step_rows["step_id"].append(steps)
        step_rows["time"].append(steps / float(env.CTRL_FREQ))
        step_rows["context"].append(context)
        step_rows["position_error"].append(current_position_error)
        step_rows["rpy"].append(drone_state[7:10])
        step_rows["angular_velocity"].append(drone_state[13:16])
        step_rows["actor_action"].append(actor_action)
        step_rows["teacher_action"].append(teacher_action)
        step_rows["applied_action"].append(applied_action)
        step_rows["actor_wrench"].append(actor_wrench)
        step_rows["teacher_wrench"].append(teacher_wrench)
        step_rows["recovery_active"].append(recovery_active_at_action)
        step_rows["recovery_early"].append(recovery_early_at_action)

        next_obs, _, terminated, truncated, info = env.step(
            applied_action.reshape(1, -1)
        )
        state = preprocess_state(next_obs.reshape(-1))
        agent.record_executed_action(applied_action)
        agent.advance_runtime_step()
        steps += 1
        next_drone_state = env._getDroneStateVector(0)
        next_reference = full_reference.sample(
            agent.runtime_step * agent.control_dt,
            lookahead_seconds=0.0,
        )
        final_position_error = float(
            np.linalg.norm(
                next_drone_state[0:3] - next_reference["current_position"]
            )
        )
        max_position_error = max(max_position_error, final_position_error)
        max_rpy = np.maximum(max_rpy, np.abs(next_drone_state[7:10]))
        max_angular_velocity = max(
            max_angular_velocity,
            float(np.linalg.norm(next_drone_state[13:16])),
        )
        if terminated or truncated:
            break

    reason = str(info.get("done_reason", "unknown"))
    action_errors = np.asarray(action_errors)
    wrench_errors = np.asarray(wrench_errors)
    result = {
        "episode_uid": episode_uid,
        "controller_mode": controller_mode,
        "scenario": scenario.name,
        "category": scenario.category,
        "reference_kind": scenario.reference_kind,
        "holdout_condition": scenario.holdout_condition,
        "seed": seed,
        "duration": duration,
        "steps": steps,
        "done_reason": reason,
        "full_horizon": reason == "timeout",
        "unstable": reason in UNSTABLE_REASONS,
        "recovery_time": recovery_time,
        "max_abs_roll": float(max_rpy[0]),
        "max_abs_pitch": float(max_rpy[1]),
        "max_abs_yaw": float(max_rpy[2]),
        "max_angular_velocity": max_angular_velocity,
        "max_position_error": max_position_error,
        "steady_state_position_error": float(
            np.mean(position_errors[-min(120, len(position_errors)) :])
        ),
        "final_position_error": final_position_error,
        "per_motor_action_mae": np.mean(np.abs(action_errors), axis=0).tolist(),
        "max_action_error": float(np.max(np.abs(action_errors))),
        "mean_abs_wrench_error": np.mean(
            np.abs(wrench_errors),
            axis=0,
        ).tolist(),
        "torque_direction_agreement": {
            name: (
                1.0
                if not matches
                else float(np.mean(matches))
            )
            for name, matches in zip(
                ("roll", "pitch", "yaw"),
                torque_direction_matches,
            )
        },
        "torque_direction_match_counts": {
            name: int(np.sum(matches))
            for name, matches in zip(
                ("roll", "pitch", "yaw"),
                torque_direction_matches,
            )
        },
        "torque_direction_eligible_counts": {
            name: int(count)
            for name, count in zip(
                ("roll", "pitch", "yaw"),
                torque_direction_eligible_counts,
            )
        },
        "first_significant_deviation": first_deviation,
        "teacher_takeover_used": False,
    }
    env.close()
    return result, step_rows


def merge_step_rows(all_rows):
    merged = {}
    for rows in all_rows:
        for name, values in rows.items():
            merged.setdefault(name, []).extend(values)
    dtypes = {
        "episode_uid": np.int32,
        "step_id": np.int32,
        "time": np.float32,
        "position_error": np.float32,
    }
    return {
        name: np.asarray(values, dtype=dtypes.get(name, np.float32))
        for name, values in merged.items()
    }


def summarize(episodes):
    unstable = [episode["unstable"] for episode in episodes]
    recoveries = [
        episode
        for episode in episodes
        if episode["category"] in {"initial_recovery", "impulse_recovery"}
    ]
    nominal_hover = [
        episode
        for episode in episodes
        if episode["scenario"] == "nominal_hover"
    ]
    direction = {}
    for axis in ("roll", "pitch", "yaw"):
        eligible = int(
            np.sum(
                [
                    episode["torque_direction_eligible_counts"][axis]
                    for episode in episodes
                ]
            )
        )
        matches = int(
            np.sum(
                [
                    episode["torque_direction_match_counts"][axis]
                    for episode in episodes
                ]
            )
        )
        direction[axis] = 1.0 if eligible == 0 else float(matches / eligible)
    return {
        "episode_count": len(episodes),
        "full_horizon_rate": float(
            np.mean([episode["full_horizon"] for episode in episodes])
        ),
        "instability_rate": float(np.mean(unstable)),
        "attitude_bound_rate": float(
            np.mean(
                [episode["done_reason"] == "attitude_bound" for episode in episodes]
            )
        ),
        "xy_bound_rate": float(
            np.mean([episode["done_reason"] == "xy_bound" for episode in episodes])
        ),
        "height_bound_rate": float(
            np.mean(
                [episode["done_reason"] == "height_bound" for episode in episodes]
            )
        ),
        "nominal_hover_success_rate": (
            float(np.mean([episode["full_horizon"] for episode in nominal_hover]))
            if nominal_hover
            else None
        ),
        "disturbance_recovery_success_rate": (
            float(
                np.mean(
                    [
                        episode["full_horizon"]
                        and episode["recovery_time"] is not None
                        for episode in recoveries
                    ]
                )
            )
            if recoveries
            else None
        ),
        "torque_direction_agreement": direction,
    }


def evaluate_gates(development_actor, formal_actor, formal_teacher):
    dev_hover = [
        episode
        for episode in development_actor
        if episode["scenario"] == "nominal_hover"
    ]
    dev_disturbance = [
        episode
        for episode in development_actor
        if episode["category"] == "initial_recovery"
    ]
    development_checks = {
        "five_hover_seeds_completed": len(dev_hover) == 5
        and all(episode["full_horizon"] for episode in dev_hover),
        "no_development_instability": all(
            not episode["unstable"] for episode in development_actor
        ),
        "small_attitude_disturbances_recovered": all(
            episode["full_horizon"] and episode["recovery_time"] is not None
            for episode in dev_disturbance
        ),
    }
    formal_summary = summarize(formal_actor)
    teacher_lookup = {
        (episode["scenario"], episode["seed"], episode["duration"]): episode
        for episode in formal_teacher
    }
    recovery_time_ratios_ok = []
    for actor_episode in formal_actor:
        if actor_episode["category"] not in {
            "initial_recovery",
            "impulse_recovery",
        }:
            continue
        teacher_episode = teacher_lookup[
            (
                actor_episode["scenario"],
                actor_episode["seed"],
                actor_episode["duration"],
            )
        ]
        actor_time = actor_episode["recovery_time"]
        teacher_time = teacher_episode["recovery_time"]
        recovery_time_ratios_ok.append(
            actor_time is not None
            and teacher_time is not None
            and actor_time <= 2.0 * teacher_time + 0.25
        )
    formal_checks = {
        "thirty_second_hover_completed": all(
            episode["full_horizon"]
            for episode in formal_actor
            if episode["scenario"] == "nominal_hover"
        ),
        "nominal_hover_success_rate_ge_95pct": (
            formal_summary["nominal_hover_success_rate"] >= 0.95
        ),
        "disturbance_recovery_success_rate_ge_90pct": (
            formal_summary["disturbance_recovery_success_rate"] >= 0.90
        ),
        "instability_rate_le_2pct": formal_summary["instability_rate"] <= 0.02,
        "recovery_time_within_teacher_multiple": all(recovery_time_ratios_ok),
        "key_torque_directions_ge_90pct": all(
            value >= 0.90
            for value in formal_summary["torque_direction_agreement"].values()
        ),
        "teacher_takeover_not_used": all(
            not episode["teacher_takeover_used"] for episode in formal_actor
        ),
    }
    return {
        "development": {
            "checks": development_checks,
            "passed": bool(all(development_checks.values())),
        },
        "formal": {
            "checks": formal_checks,
            "passed": bool(all(formal_checks.values())),
        },
        "actor_hover_gate_passed": bool(
            development_checks["five_hover_seeds_completed"]
            and formal_checks["thirty_second_hover_completed"]
            and formal_checks["nominal_hover_success_rate_ge_95pct"]
            and formal_checks["instability_rate_le_2pct"]
        ),
        "actor_disturbance_recovery_gate_passed": bool(
            development_checks["small_attitude_disturbances_recovered"]
            and formal_checks["disturbance_recovery_success_rate_ge_90pct"]
            and formal_checks["recovery_time_within_teacher_multiple"]
            and formal_checks["key_torque_directions_ge_90pct"]
            and formal_checks["instability_rate_le_2pct"]
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Independent closed-loop gate for the behavior-cloned motor actor."
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/behavior_cloning/"
            "asymmetric_rpm_v2_plain_mlp_b4096/actor_best.pt"
        ),
    )
    parser.add_argument(
        "--dataset-metadata",
        default="data/behavior_cloning/asymmetric_rpm_v2/metadata.json",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/behavior_cloning/asymmetric_rpm_v2_plain_mlp_b4096",
    )
    args = parser.parse_args()
    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "closed_loop_evaluation.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite {result_path}.")

    metadata = json.loads(
        pathlib.Path(args.dataset_metadata).read_text(encoding="utf-8")
    )
    assert metadata["motor_action_codec"] == "asymmetric_rpm"
    assert metadata["teacher_reset_enabled"] is True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if checkpoint["dataset_samples_sha256"] != metadata["samples_sha256"]:
        raise ValueError("Checkpoint and dataset metadata hashes do not match.")
    physical_parameters = metadata["motor_physical_parameters"]
    torque_thresholds = metadata["large_torque_thresholds"]
    critical_torque_thresholds = metadata.get(
        "critical_recovery_torque_thresholds",
        torque_thresholds,
    )

    development_actor = []
    formal_actor = []
    formal_teacher = []
    all_step_rows = []
    episode_uid = 0
    for scenario, seed, duration in development_episodes():
        result, rows = run_episode(
            scenario,
            seed,
            duration,
            controller_mode="actor",
            checkpoint=checkpoint,
            device=device,
            physical_parameters=physical_parameters,
            torque_thresholds=torque_thresholds,
            critical_torque_thresholds=critical_torque_thresholds,
            episode_uid=episode_uid,
        )
        development_actor.append(result)
        all_step_rows.append(rows)
        episode_uid += 1
        print(
            f"[bc-eval] development {len(development_actor)}/"
            f"{len(development_episodes())}: {scenario.name} "
            f"{result['done_reason']} steps={result['steps']}",
            file=sys.stderr,
            flush=True,
        )

    for scenario, seed, duration in formal_episodes():
        teacher_result, _ = run_episode(
            scenario,
            seed,
            duration,
            controller_mode="teacher",
            checkpoint=checkpoint,
            device=device,
            physical_parameters=physical_parameters,
            torque_thresholds=torque_thresholds,
            critical_torque_thresholds=critical_torque_thresholds,
            episode_uid=episode_uid,
        )
        formal_teacher.append(teacher_result)
        actor_result, rows = run_episode(
            scenario,
            seed,
            duration,
            controller_mode="actor",
            checkpoint=checkpoint,
            device=device,
            physical_parameters=physical_parameters,
            torque_thresholds=torque_thresholds,
            critical_torque_thresholds=critical_torque_thresholds,
            episode_uid=episode_uid,
        )
        formal_actor.append(actor_result)
        all_step_rows.append(rows)
        episode_uid += 1
        print(
            f"[bc-eval] formal {len(formal_actor)}/{len(formal_episodes())}: "
            f"{scenario.name} actor={actor_result['done_reason']} "
            f"teacher={teacher_result['done_reason']}",
            file=sys.stderr,
            flush=True,
        )

    gates = evaluate_gates(development_actor, formal_actor, formal_teacher)
    result = {
        "checkpoint": args.checkpoint,
        "dataset_version": metadata["dataset_version"],
        "motor_action_codec": metadata["motor_action_codec"],
        "teacher_parallel_diagnostic_only": True,
        "teacher_takeover_used": False,
        "critical_recovery_torque_thresholds": critical_torque_thresholds,
        "development_summary": summarize(development_actor),
        "formal_actor_summary": summarize(formal_actor),
        "formal_teacher_summary": summarize(formal_teacher),
        "gates": gates,
        "development_actor_episodes": development_actor,
        "formal_actor_episodes": formal_actor,
        "formal_teacher_episodes": formal_teacher,
    }
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output / "step_diagnostics.npz",
        **merge_step_rows(all_step_rows),
    )
    print(json.dumps(gates, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
