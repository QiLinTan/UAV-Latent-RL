from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
from collections import Counter

import numpy as np

from utils.gym_pybullet_compat import ensure_gym_pybullet_envs_compat

ensure_gym_pybullet_envs_compat()

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from algos.td3.td3_reference_tracking import TD3ReferenceTracking
from data.behavior_cloning_dataset import EXPECTED_DATASET_VERSION
from envs.preprocess import preprocess_state
from models.motor_action_codec import ASYMMETRIC_RPM
from scripts.bc_scenarios import (
    GROUP_NAMES,
    apply_initial_disturbance,
    apply_runtime_impulse,
    build_scenarios,
    disturbance_vector,
    impulse_step,
    make_env,
    make_reference_packet,
    make_reference_window,
    physical_state,
    recovery_state,
)


def git_state(path: pathlib.Path):
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


def motor_wrench(rpm, *, kf: float, km: float, arm_length: float):
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
        dtype=np.float64,
    )


def make_agent(env):
    return TD3ReferenceTracking(
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
    )


def assert_teacher_reset(controller):
    controller.reset()
    np.testing.assert_allclose(controller.integral_pos_e, 0.0)
    np.testing.assert_allclose(controller.integral_rpy_e, 0.0)
    np.testing.assert_allclose(controller.last_rpy, 0.0)
    np.testing.assert_allclose(controller.last_rpy_e, 0.0)
    assert controller.control_counter == 0


def split_name(scenario, seed: int):
    if scenario.holdout_condition:
        return "test_unseen_condition"
    if seed in (0, 1, 2):
        return "train"
    if seed == 3:
        return "validation"
    if seed == 4:
        return "test_unseen_seed"
    raise ValueError(f"No split policy for seed {seed}.")


def main():
    parser = argparse.ArgumentParser(
        description="Collect versioned asymmetric-RPM teacher behavior-cloning data."
    )
    parser.add_argument(
        "--output-dir",
        default="data/behavior_cloning/asymmetric_rpm_v2",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episode-len-sec", type=float, default=12.0)
    args = parser.parse_args()
    if sorted(args.seeds) != [0, 1, 2, 3, 4]:
        raise ValueError("The v2 split contract requires exactly seeds 0 1 2 3 4.")

    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    protected_outputs = [
        output / "metadata.json",
        output / "samples.npz",
        output / "splits.npz",
        output / "episode_manifest.jsonl",
    ]
    existing = [path for path in protected_outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite an existing dataset: "
            + ", ".join(str(path) for path in existing)
        )

    scenarios = build_scenarios()
    disturbance_names = {scenario.name: index for index, scenario in enumerate(scenarios)}
    arrays = {
        "episode_id": [],
        "step_id": [],
        "time": [],
        "state": [],
        "context": [],
        "current_reference": [],
        "lookahead_reference": [],
        "reference_age": [],
        "reference_valid": [],
        "previous_action": [],
        "raw_teacher_rpm": [],
        "encoded_teacher_action": [],
        "applied_action": [],
        "next_state": [],
        "disturbance_type_id": [],
        "disturbance_magnitude": [],
        "recovery_flag": [],
        "recovery_early_flag": [],
        "teacher_wrench": [],
        "sample_group_id": [],
    }
    manifest = []
    split_episode_ids = {
        "train": [],
        "validation": [],
        "test_unseen_seed": [],
        "test_unseen_condition": [],
    }
    physical_saturated_motors = 0
    action_motor_count = 0
    rpm_roundtrip_absolute_error_sum = 0.0
    rpm_roundtrip_error_count = 0
    max_rpm_roundtrip_error = 0.0
    motor_physical_parameters = None
    episode_id = 0
    total_episodes = len(scenarios) * len(args.seeds)

    for scenario in scenarios:
        for seed in args.seeds:
            env = make_env(scenario, float(args.episode_len_sec))
            if motor_physical_parameters is None:
                motor_physical_parameters = {
                    "min_rpm": 0.0,
                    "hover_rpm": float(env.HOVER_RPM),
                    "max_rpm": float(env.MAX_RPM),
                    "kf": float(env.KF),
                    "km": float(env.KM),
                    "arm_length": float(env.L),
                }
            obs, _ = env.reset(seed=seed)
            apply_initial_disturbance(env, scenario, seed)
            obs = env._computeObs()
            state = preprocess_state(obs.reshape(-1))
            agent = make_agent(env)
            agent.reset_episode()
            codec = agent.configure_motor_action_interface(env)
            agent.set_high_level_enabled(False)
            teacher = DSLPIDControl(drone_model=env.DRONE_MODEL)
            assert_teacher_reset(teacher)
            full_reference = make_reference_packet(
                scenario.reference_kind,
                float(args.episode_len_sec),
            )
            disturbance = disturbance_vector(scenario, seed)
            event_step = impulse_step(scenario, env.CTRL_FREQ)
            disturbance_started = scenario.category == "initial_recovery"
            recovery_active = disturbance_started
            recovery_start = 0 if disturbance_started else None
            stable_count = 0
            recovery_time = None
            steps = 0
            info = {}
            max_position_error = 0.0
            max_attitude = 0.0
            reference_version = 0

            while True:
                if event_step is not None and steps == event_step:
                    apply_runtime_impulse(env, scenario, seed)
                    obs = env._computeObs()
                    state = preprocess_state(obs.reshape(-1))
                    disturbance_started = True
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
                    packet = agent.reference_buffer.packet
                    raise RuntimeError(
                        "Teacher collection encountered an invalid reference: "
                        f"scenario={scenario.name}, seed={seed}, step={steps}, "
                        f"now={steps / float(env.CTRL_FREQ):.9f}, "
                        f"packet_start={None if packet is None else packet.t_start}, "
                        f"packet_expires={None if packet is None else packet.expires_at}, "
                        f"sample={reference_sample}."
                    )

                stable_now = recovery_state(env, reference_sample)
                if recovery_active:
                    stable_count = stable_count + 1 if stable_now else 0
                    if stable_count >= 30:
                        first_stable_step = steps - stable_count + 1
                        recovery_time = (
                            first_stable_step - int(recovery_start)
                        ) / float(env.CTRL_FREQ)
                        recovery_active = False

                previous_action = agent.previous_action.copy()
                drone_state = env._getDroneStateVector(0)
                raw_rpm, _, _ = teacher.computeControl(
                    control_timestep=env.CTRL_TIMESTEP,
                    cur_pos=drone_state[0:3],
                    cur_quat=drone_state[3:7],
                    cur_vel=drone_state[10:13],
                    cur_ang_vel=drone_state[13:16],
                    target_pos=reference_sample["lookahead_position"],
                    target_vel=reference_sample["lookahead_velocity"],
                )
                encoded_action = codec.rpm_to_normalized_action(raw_rpm).astype(
                    np.float32
                )
                applied_action = encoded_action.copy()
                if not np.array_equal(encoded_action, applied_action):
                    raise AssertionError(
                        "encoded_teacher_action must exactly equal applied_action."
                    )
                reconstructed_rpm = codec.normalized_action_to_rpm(applied_action)
                saturation = codec.physical_constraint.saturation_mask(raw_rpm, atol=1e-6)
                physical_saturated_motors += int(np.sum(saturation))
                action_motor_count += 4
                rpm_roundtrip_error = np.abs(reconstructed_rpm - raw_rpm)
                rpm_roundtrip_absolute_error_sum += float(
                    np.sum(rpm_roundtrip_error)
                )
                rpm_roundtrip_error_count += int(rpm_roundtrip_error.size)
                max_rpm_roundtrip_error = max(
                    max_rpm_roundtrip_error,
                    float(np.max(rpm_roundtrip_error)),
                )
                if not np.all(saturation) and np.max(rpm_roundtrip_error) > 1e-2:
                    raise AssertionError(
                        "Float32 T3 execution changed an unsaturated teacher RPM "
                        f"by more than 0.01 RPM: max={np.max(rpm_roundtrip_error)}."
                    )

                current_reference = np.concatenate(
                    [
                        reference_sample["current_position"],
                        reference_sample["current_velocity"],
                    ]
                ).astype(np.float32)
                lookahead_reference = np.concatenate(
                    [
                        reference_sample["lookahead_position"],
                        reference_sample["lookahead_velocity"],
                    ]
                ).astype(np.float32)
                current_physical_state = physical_state(env)
                wrench = motor_wrench(
                    raw_rpm,
                    kf=env.KF,
                    km=env.KM,
                    arm_length=env.L,
                ).astype(np.float32)

                next_obs, _, terminated, truncated, info = env.step(
                    applied_action.reshape(1, -1)
                )
                next_state_physical = physical_state(env)
                next_state = preprocess_state(next_obs.reshape(-1))

                arrays["episode_id"].append(episode_id)
                arrays["step_id"].append(steps)
                arrays["time"].append(steps / float(env.CTRL_FREQ))
                arrays["state"].append(current_physical_state)
                arrays["context"].append(context)
                arrays["current_reference"].append(current_reference)
                arrays["lookahead_reference"].append(lookahead_reference)
                arrays["reference_age"].append(reference_sample["age_seconds"])
                arrays["reference_valid"].append(reference_sample["valid"])
                arrays["previous_action"].append(previous_action)
                arrays["raw_teacher_rpm"].append(raw_rpm)
                arrays["encoded_teacher_action"].append(encoded_action)
                arrays["applied_action"].append(applied_action)
                arrays["next_state"].append(next_state_physical)
                arrays["disturbance_type_id"].append(disturbance_names[scenario.name])
                arrays["disturbance_magnitude"].append(disturbance)
                arrays["recovery_flag"].append(recovery_active)
                arrays["recovery_early_flag"].append(
                    recovery_active
                    and recovery_start is not None
                    and steps - int(recovery_start) < 60
                )
                arrays["teacher_wrench"].append(wrench)
                arrays["sample_group_id"].append(scenario.group_id)

                current_position_error = float(
                    np.linalg.norm(
                        env.pos[0] - reference_sample["current_position"]
                    )
                )
                max_position_error = max(max_position_error, current_position_error)
                max_attitude = max(
                    max_attitude,
                    abs(float(env.rpy[0][0])),
                    abs(float(env.rpy[0][1])),
                    abs(float(env.rpy[0][2])),
                )
                agent.record_executed_action(applied_action)
                agent.advance_runtime_step()
                state = next_state
                steps += 1
                if terminated or truncated:
                    break

            done_reason = str(info.get("done_reason", "unknown"))
            if done_reason != "timeout":
                raise RuntimeError(
                    f"Teacher failed dataset scenario {scenario.name!r}, seed {seed}: "
                    f"{done_reason} at step {steps}."
                )
            split = split_name(scenario, seed)
            split_episode_ids[split].append(episode_id)
            manifest.append(
                {
                    "episode_id": episode_id,
                    "scenario": scenario.name,
                    "category": scenario.category,
                    "reference_kind": scenario.reference_kind,
                    "random_seed": seed,
                    "split": split,
                    "holdout_condition": scenario.holdout_condition,
                    "steps": steps,
                    "done_reason": done_reason,
                    "recovery_time": recovery_time,
                    "max_position_error": max_position_error,
                    "max_attitude": max_attitude,
                    "disturbance_magnitude": disturbance.tolist(),
                }
            )
            env.close()
            episode_id += 1
            if episode_id % 10 == 0 or episode_id == total_episodes:
                print(
                    f"[collect-bc] completed {episode_id}/{total_episodes}",
                    file=sys.stderr,
                    flush=True,
                )

    scalar_dtypes = {
        "episode_id": np.int32,
        "step_id": np.int32,
        "time": np.float32,
        "reference_age": np.float32,
        "reference_valid": np.bool_,
        "disturbance_type_id": np.int16,
        "recovery_flag": np.bool_,
        "recovery_early_flag": np.bool_,
        "sample_group_id": np.int8,
    }
    samples = {
        name: np.asarray(values, dtype=scalar_dtypes.get(name, np.float32))
        for name, values in arrays.items()
    }
    train_episode_ids = np.asarray(split_episode_ids["train"], dtype=np.int32)
    train_mask = np.isin(samples["episode_id"], train_episode_ids)
    absolute_torques = np.abs(samples["teacher_wrench"][train_mask, 1:4])
    torque_thresholds = np.percentile(absolute_torques, 90.0, axis=0)
    torque_thresholds = np.maximum(torque_thresholds, 1e-10)
    early_recovery_mask = np.logical_and(
        train_mask,
        samples["recovery_early_flag"],
    )
    critical_recovery_torque_thresholds = np.percentile(
        np.abs(samples["teacher_wrench"][early_recovery_mask, 1:4]),
        75.0,
        axis=0,
    )
    critical_recovery_torque_thresholds = np.maximum(
        critical_recovery_torque_thresholds,
        1e-10,
    )
    samples["large_roll_torque_flag"] = (
        np.abs(samples["teacher_wrench"][:, 1]) >= torque_thresholds[0]
    )
    samples["large_pitch_torque_flag"] = (
        np.abs(samples["teacher_wrench"][:, 2]) >= torque_thresholds[1]
    )
    samples["large_yaw_torque_flag"] = (
        np.abs(samples["teacher_wrench"][:, 3]) >= torque_thresholds[2]
    )

    np.savez_compressed(output / "samples.npz", **samples)
    np.savez(
        output / "splits.npz",
        **{
            name: np.asarray(ids, dtype=np.int32)
            for name, ids in split_episode_ids.items()
        },
    )
    (output / "episode_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest),
        encoding="utf-8",
    )

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    environment_root = repo_root.parent / "gym-pybullet-drones"
    project_state = git_state(repo_root)
    environment_state = git_state(environment_root)
    split_counts = {
        split: {
            "episodes": len(ids),
            "samples": int(
                np.sum(np.isin(samples["episode_id"], np.asarray(ids, dtype=np.int32)))
            ),
        }
        for split, ids in split_episode_ids.items()
    }
    group_counts = {
        GROUP_NAMES[group_id]: int(np.sum(samples["sample_group_id"] == group_id))
        for group_id in GROUP_NAMES
    }
    scenario_counts = Counter(
        row["scenario"] for row in manifest
    )
    samples_sha256 = hashlib.sha256(
        (output / "samples.npz").read_bytes()
    ).hexdigest()
    metadata = {
        "dataset_version": EXPECTED_DATASET_VERSION,
        "motor_action_codec": ASYMMETRIC_RPM,
        "teacher_controller": (
            "gym_pybullet_drones.control.DSLPIDControl.DSLPIDControl"
        ),
        "teacher_reset_enabled": True,
        "teacher_action_noise": 0.0,
        "encoded_action_equals_applied_action": bool(
            np.array_equal(
                samples["encoded_teacher_action"],
                samples["applied_action"],
            )
        ),
        "control_frequency": 120,
        "physics_frequency": 240,
        "reference_mode": sorted(
            {scenario.reference_kind for scenario in scenarios}
        ),
        "reference_update_interval_steps": 8,
        "reference_horizon_seconds": 1.0,
        "random_seed": list(args.seeds),
        "environment_commit": environment_state["commit"],
        "environment_dirty": environment_state["dirty"],
        "project_commit": project_state["commit"],
        "project_dirty": project_state["dirty"],
        "context_dim": int(samples["context"].shape[1]),
        "context_definition": (
            "normalized kinematics[12] + current position/velocity error[6] + "
            "lookahead position/velocity error[6] + reference age/valid[2] + "
            "previous applied action[4]; no multi-step action history"
        ),
        "sample_count": int(samples["episode_id"].shape[0]),
        "episode_count": int(len(manifest)),
        "split_counts": split_counts,
        "sample_group_counts": group_counts,
        "scenario_episode_counts": dict(sorted(scenario_counts.items())),
        "disturbance_type_mapping": disturbance_names,
        "disturbance_magnitude_layout": (
            "initial_rpy_xyz, linear_velocity_xyz, angular_velocity_xyz, height_offset"
        ),
        "recovery_flag_definition": (
            "from initial/runtime disturbance until 30 consecutive steps satisfy "
            "position<=0.05m, attitude<=0.05rad, velocity<=0.15m/s, "
            "angular_velocity<=0.15rad/s"
        ),
        "large_torque_percentile": 90.0,
        "large_torque_thresholds": {
            "roll": float(torque_thresholds[0]),
            "pitch": float(torque_thresholds[1]),
            "yaw": float(torque_thresholds[2]),
        },
        "critical_recovery_torque_definition": (
            "75th percentile absolute teacher torque among training-split "
            "recovery_early samples; used only for online recovery-direction gates"
        ),
        "critical_recovery_torque_thresholds": {
            "roll": float(critical_recovery_torque_thresholds[0]),
            "pitch": float(critical_recovery_torque_thresholds[1]),
            "yaw": float(critical_recovery_torque_thresholds[2]),
        },
        "physical_saturation_motor_fraction": float(
            physical_saturated_motors / max(1, action_motor_count)
        ),
        "motor_physical_parameters": motor_physical_parameters,
        "float32_rpm_roundtrip_mean_absolute_error": float(
            rpm_roundtrip_absolute_error_sum / max(1, rpm_roundtrip_error_count)
        ),
        "float32_rpm_roundtrip_max_absolute_error": float(
            max_rpm_roundtrip_error
        ),
        "samples_sha256": samples_sha256,
        "quarantined": False,
        "training_allowed": True,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
