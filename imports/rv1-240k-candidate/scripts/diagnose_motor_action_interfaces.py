from __future__ import annotations

import argparse
import contextlib
import io
import json
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import pybullet as p

from utils.gym_pybullet_compat import ensure_gym_pybullet_envs_compat

ensure_gym_pybullet_envs_compat()

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from envs.ForestAviary import CustomForestAviary
from models.motor_action_codec import (
    ASYMMETRIC_RPM,
    ASYMMETRIC_THRUST,
    LegacyFixedScaleMotorActionCodec,
    MotorActionCodec,
    MotorPhysicalLimits,
)
from models.reference_packet import ActuatorConstraintLayer, ReferencePacket


INTERFACES = {
    "T1_raw_dslpid_rpm": "raw",
    "T2_legacy_projected": "legacy",
    "T3_asymmetric_rpm": ASYMMETRIC_RPM,
    "T4_asymmetric_thrust": ASYMMETRIC_THRUST,
}
UNSTABLE_REASONS = {"attitude_bound", "height_bound", "collision", "xy_bound"}


@dataclass(frozen=True)
class DisturbanceCase:
    name: str
    reference: str = "hover"
    initial_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_angular_velocity: tuple[float, float, float] | None = None
    height_offset: float = 0.0
    runtime_impulse: bool = False

    @property
    def has_initial_disturbance(self) -> bool:
        return bool(
            np.max(np.abs(self.initial_rpy)) > 0.0
            or np.max(np.abs(self.initial_velocity)) > 0.0
            or self.initial_angular_velocity is not None
            or abs(self.height_offset) > 0.0
        )


def build_cases() -> list[DisturbanceCase]:
    cases = [DisturbanceCase("hover_zero")]
    axes = ("roll", "pitch", "yaw")
    for axis_index, axis in enumerate(axes):
        for amplitude in (-0.20, -0.10, -0.05, 0.05, 0.10, 0.20):
            rpy = [0.0, 0.0, 0.0]
            rpy[axis_index] = amplitude
            cases.append(
                DisturbanceCase(
                    f"initial_{axis}_{amplitude:+.2f}",
                    initial_rpy=tuple(rpy),
                )
            )
    for amplitude in (-0.20, -0.10, -0.05, 0.05, 0.10, 0.20):
        cases.append(
            DisturbanceCase(
                f"initial_rpy_mixed_{amplitude:+.2f}",
                initial_rpy=(amplitude, -0.8 * amplitude, 0.6 * amplitude),
            )
        )
    cases.append(
        DisturbanceCase(
            "initial_rpy_regression_0.12_-0.10_0.08",
            initial_rpy=(0.12, -0.10, 0.08),
        )
    )
    cases.extend(
        [
            DisturbanceCase("initial_angular_velocity_random"),
            DisturbanceCase("initial_velocity_x_pos", initial_velocity=(0.4, 0.0, 0.0)),
            DisturbanceCase("initial_velocity_x_neg", initial_velocity=(-0.4, 0.0, 0.0)),
            DisturbanceCase("initial_velocity_y_pos", initial_velocity=(0.0, 0.4, 0.0)),
            DisturbanceCase("initial_velocity_y_neg", initial_velocity=(0.0, -0.4, 0.0)),
            DisturbanceCase("height_offset_-0.20", height_offset=-0.20),
            DisturbanceCase("height_offset_-0.10", height_offset=-0.10),
            DisturbanceCase("height_offset_+0.10", height_offset=0.10),
            DisturbanceCase("height_offset_+0.20", height_offset=0.20),
            DisturbanceCase("runtime_velocity_angular_impulse", runtime_impulse=True),
            DisturbanceCase("straight_reference", reference="line"),
            DisturbanceCase("gentle_curve_reference", reference="curve"),
        ]
    )
    return cases


def make_reference_packet(kind: str, duration: float) -> ReferencePacket:
    timestamps = np.linspace(0.0, duration + 0.1, 123, dtype=np.float64)
    normalized_time = np.clip(timestamps / duration, 0.0, 1.0)
    smooth = (
        10.0 * normalized_time**3
        - 15.0 * normalized_time**4
        + 6.0 * normalized_time**5
    )
    smooth_rate = (
        30.0 * normalized_time**2
        - 60.0 * normalized_time**3
        + 30.0 * normalized_time**4
    ) / duration

    positions = np.zeros((timestamps.size, 3), dtype=np.float64)
    velocities = np.zeros_like(positions)
    if kind == "line":
        positions[:, 0] = 1.8 * smooth
        velocities[:, 0] = 1.8 * smooth_rate
    elif kind == "curve":
        positions[:, 0] = 1.5 * smooth
        positions[:, 1] = 0.25 * np.sin(np.pi * smooth)
        velocities[:, 0] = 1.5 * smooth_rate
        velocities[:, 1] = 0.25 * np.pi * np.cos(np.pi * smooth) * smooth_rate
    elif kind != "hover":
        raise ValueError(f"Unknown reference type {kind!r}.")

    return ReferencePacket(
        positions=positions.astype(np.float32),
        velocities=velocities.astype(np.float32),
        relative_timestamps=timestamps,
        t_gen=0.0,
        t_start=0.0,
        t_receive=0.0,
        valid_duration=duration + 0.1,
        version=1,
        frame_id="local_navigation",
        origin_position=np.array([-3.5, 0.0, 1.0], dtype=np.float32),
        origin_attitude=np.zeros(3, dtype=np.float32),
    )


def make_env(case: DisturbanceCase, episode_len_sec: float):
    initial_xyz = np.array(
        [[-3.5, 0.0, 1.0 + case.height_offset]],
        dtype=np.float32,
    )
    initial_rpy = np.array([case.initial_rpy], dtype=np.float32)
    with contextlib.redirect_stdout(io.StringIO()):
        return CustomForestAviary(
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            initial_xyzs=initial_xyz,
            initial_rpys=initial_rpy,
            pyb_freq=240,
            ctrl_freq=120,
            gui=False,
            curriculum=False,
            num_trees=0,
            route_blocking_tree=False,
            target_pos=[3.5, 0.0, 1.0],
            episode_len_sec=episode_len_sec,
        )


def motor_wrench(rpm, *, kf: float, km: float, arm_length: float) -> np.ndarray:
    rpm = np.asarray(rpm, dtype=np.float64)
    forces = float(kf) * np.square(rpm)
    squared = np.square(rpm)
    total_thrust = float(np.sum(forces))
    roll = float((forces[0] + forces[1] - forces[2] - forces[3]) * arm_length / np.sqrt(2.0))
    pitch = float((-forces[0] + forces[1] + forces[2] - forces[3]) * arm_length / np.sqrt(2.0))
    yaw = float((-squared[0] + squared[1] - squared[2] + squared[3]) * km)
    return np.array([total_thrust, roll, pitch, yaw], dtype=np.float64)


def transform_teacher_rpm(
    raw_rpm,
    interface: str,
    *,
    legacy_codec: LegacyFixedScaleMotorActionCodec,
    legacy_constraint: ActuatorConstraintLayer,
    new_codec: MotorActionCodec | None,
) -> tuple[np.ndarray, np.ndarray]:
    raw_rpm = np.asarray(raw_rpm, dtype=np.float64)
    if interface == "raw":
        action = legacy_codec.rpm_to_normalized_action(raw_rpm)
        return action, raw_rpm.copy()
    if interface == "legacy":
        action = legacy_constraint(
            legacy_codec.rpm_to_normalized_action(raw_rpm)
        )
        return action, legacy_codec.normalized_action_to_rpm(action)
    action = new_codec.rpm_to_normalized_action(raw_rpm)
    return action, new_codec.normalized_action_to_rpm(action)


def apply_initial_velocity(env, case: DisturbanceCase, seed: int):
    angular_velocity = case.initial_angular_velocity
    if case.name == "initial_angular_velocity_random":
        angular_velocity = tuple(
            np.random.default_rng(seed).uniform(-0.45, 0.45, size=3)
        )
    if np.max(np.abs(case.initial_velocity)) > 0.0 or angular_velocity is not None:
        p.resetBaseVelocity(
            env.DRONE_IDS[0],
            linearVelocity=case.initial_velocity,
            angularVelocity=(0.0, 0.0, 0.0) if angular_velocity is None else angular_velocity,
            physicsClientId=env.CLIENT,
        )
        env._updateAndStoreKinematicInformation()
    return angular_velocity


def episode(
    interface_name: str,
    interface: str,
    case: DisturbanceCase,
    seed: int,
    *,
    episode_len_sec: float,
    collect_samples: bool,
):
    env = make_env(case, episode_len_sec)
    env.reset(seed=seed)
    angular_velocity = apply_initial_velocity(env, case, seed)
    controller = DSLPIDControl(drone_model=env.DRONE_MODEL)
    controller.reset()
    reference = make_reference_packet(case.reference, episode_len_sec)
    limits = MotorPhysicalLimits(
        min_rpm=0.0,
        hover_rpm=float(env.HOVER_RPM),
        max_rpm=float(env.MAX_RPM),
        kf=float(env.KF),
    )
    legacy_codec = LegacyFixedScaleMotorActionCodec(env.HOVER_RPM)
    legacy_constraint = ActuatorConstraintLayer(4, 0.75, 0.60, 0.25)
    new_codec = (
        None
        if interface in {"raw", "legacy"}
        else MotorActionCodec(limits, mode=interface)
    )
    env.set_motor_action_codec(new_codec)

    steps = 0
    impulse_step = 2 * env.CTRL_FREQ
    disturbance_start_step = impulse_step if case.runtime_impulse else 0
    stable_steps = 0
    recovery_time = None
    max_position_error = 0.0
    max_rpy = np.zeros(3)
    max_angular_velocity = 0.0
    rpm_min = np.inf
    rpm_max = -np.inf
    physical_saturated_motors = 0
    modified_steps = 0
    raw_wrench_values = []
    executed_wrench_values = []
    sample_rows = []
    info = {}

    while True:
        if case.runtime_impulse and steps == impulse_step:
            p.resetBaseVelocity(
                env.DRONE_IDS[0],
                linearVelocity=[0.6, -0.5, 0.25],
                angularVelocity=[0.8, -0.6, 0.4],
                physicsClientId=env.CLIENT,
            )
            env._updateAndStoreKinematicInformation()

        now = steps / float(env.CTRL_FREQ)
        sample = reference.sample(now, lookahead_seconds=0.15)
        state = env._getDroneStateVector(0)
        raw_rpm, _, _ = controller.computeControl(
            control_timestep=env.CTRL_TIMESTEP,
            cur_pos=state[0:3],
            cur_quat=state[3:7],
            cur_vel=state[10:13],
            cur_ang_vel=state[13:16],
            target_pos=sample["lookahead_position"],
            target_vel=sample["lookahead_velocity"],
        )
        action, executed_rpm = transform_teacher_rpm(
            raw_rpm,
            interface,
            legacy_codec=legacy_codec,
            legacy_constraint=legacy_constraint,
            new_codec=new_codec,
        )
        if np.max(np.abs(executed_rpm - raw_rpm)) > 1e-6:
            modified_steps += 1
        physical_saturated_motors += int(
            np.sum(
                np.logical_or(
                    raw_rpm <= limits.min_rpm + 1e-6,
                    raw_rpm >= limits.max_rpm - 1e-6,
                )
            )
        )
        rpm_min = min(rpm_min, float(np.min(executed_rpm)))
        rpm_max = max(rpm_max, float(np.max(executed_rpm)))
        raw_wrench = motor_wrench(
            raw_rpm,
            kf=env.KF,
            km=env.KM,
            arm_length=env.L,
        )
        executed_wrench = motor_wrench(
            executed_rpm,
            kf=env.KF,
            km=env.KM,
            arm_length=env.L,
        )
        raw_wrench_values.append(raw_wrench)
        executed_wrench_values.append(executed_wrench)
        if collect_samples:
            sample_rows.append(
                {
                    "case": case.name,
                    "raw_rpm": np.asarray(raw_rpm, dtype=np.float64),
                    "raw_wrench": raw_wrench,
                    "limits": limits,
                    "kf": float(env.KF),
                    "km": float(env.KM),
                    "arm_length": float(env.L),
                }
            )

        obs, _, terminated, truncated, info = env.step(action.reshape(1, -1))
        del obs
        steps += 1
        state = env._getDroneStateVector(0)
        current_sample = reference.sample(
            steps / float(env.CTRL_FREQ),
            lookahead_seconds=0.0,
        )
        position_error = float(
            np.linalg.norm(state[0:3] - current_sample["current_position"])
        )
        velocity_error = float(
            np.linalg.norm(state[10:13] - current_sample["current_velocity"])
        )
        rpy_abs = np.abs(state[7:10])
        angular_speed = float(np.linalg.norm(state[13:16]))
        max_position_error = max(max_position_error, position_error)
        max_rpy = np.maximum(max_rpy, rpy_abs)
        max_angular_velocity = max(max_angular_velocity, angular_speed)

        if steps >= disturbance_start_step:
            stable = bool(
                position_error <= 0.05
                and np.max(rpy_abs) <= 0.05
                and velocity_error <= 0.15
                and angular_speed <= 0.15
            )
            stable_steps = stable_steps + 1 if stable else 0
            if recovery_time is None and stable_steps >= 30:
                first_stable_step = steps - stable_steps + 1
                recovery_time = max(
                    0.0,
                    (first_stable_step - disturbance_start_step) / float(env.CTRL_FREQ),
                )

        if terminated or truncated:
            break

    raw_wrench_values = np.asarray(raw_wrench_values)
    executed_wrench_values = np.asarray(executed_wrench_values)
    wrench_error = executed_wrench_values - raw_wrench_values
    final_sample = reference.sample(
        steps / float(env.CTRL_FREQ),
        lookahead_seconds=0.0,
    )
    final_state = env._getDroneStateVector(0)
    final_position_error = float(
        np.linalg.norm(final_state[0:3] - final_sample["current_position"])
    )
    reason = str(info.get("done_reason", "unknown"))
    full_horizon = reason == "timeout"
    result = {
        "interface": interface_name,
        "case": case.name,
        "reference": case.reference,
        "seed": int(seed),
        "initial_rpy": list(case.initial_rpy),
        "initial_velocity": list(case.initial_velocity),
        "initial_angular_velocity": (
            None if angular_velocity is None else list(angular_velocity)
        ),
        "height_offset": float(case.height_offset),
        "runtime_impulse": bool(case.runtime_impulse),
        "steps": int(steps),
        "seconds": float(steps / env.CTRL_FREQ),
        "done_reason": reason,
        "full_horizon": full_horizon,
        "unstable": reason in UNSTABLE_REASONS,
        "max_abs_roll": float(max_rpy[0]),
        "max_abs_pitch": float(max_rpy[1]),
        "max_abs_yaw": float(max_rpy[2]),
        "max_angular_velocity": float(max_angular_velocity),
        "max_position_error": float(max_position_error),
        "recovery_time": recovery_time,
        "final_position_error": final_position_error,
        "rpm_min": float(rpm_min),
        "rpm_max": float(rpm_max),
        "physical_saturation_fraction": float(
            physical_saturated_motors / max(1, 4 * steps)
        ),
        "action_modified_fraction": float(modified_steps / max(1, steps)),
        "mean_abs_total_thrust_error": float(np.mean(np.abs(wrench_error[:, 0]))),
        "mean_abs_roll_torque_error": float(np.mean(np.abs(wrench_error[:, 1]))),
        "mean_abs_pitch_torque_error": float(np.mean(np.abs(wrench_error[:, 2]))),
        "mean_abs_yaw_torque_error": float(np.mean(np.abs(wrench_error[:, 3]))),
    }
    env.close()
    return result, sample_rows


def interface_transform(
    raw_rpm,
    interface: str,
    limits: MotorPhysicalLimits,
):
    legacy_codec = LegacyFixedScaleMotorActionCodec(limits.hover_rpm)
    if interface == "raw":
        return np.asarray(raw_rpm, dtype=np.float64)
    if interface == "legacy":
        constraint = ActuatorConstraintLayer(4, 0.75, 0.60, 0.25)
        action = constraint(legacy_codec.rpm_to_normalized_action(raw_rpm))
        return legacy_codec.normalized_action_to_rpm(action)
    codec = MotorActionCodec(limits, mode=interface)
    return codec.normalized_action_to_rpm(
        codec.rpm_to_normalized_action(raw_rpm)
    )


def summarize_offline_samples(sample_rows):
    raw_rpm = np.stack([row["raw_rpm"] for row in sample_rows])
    raw_wrench = np.stack([row["raw_wrench"] for row in sample_rows])
    case_names = np.asarray([row["case"] for row in sample_rows])
    limits = sample_rows[0]["limits"]
    physical_saturated = np.logical_or(
        raw_rpm <= limits.min_rpm + 1e-6,
        raw_rpm >= limits.max_rpm - 1e-6,
    )
    unsaturated_samples = ~np.any(physical_saturated, axis=1)
    recovery_roll_samples = np.char.startswith(case_names.astype(str), "initial_roll")
    recovery_roll_samples |= case_names == "runtime_velocity_angular_impulse"
    summaries = {}

    for interface_name, interface in INTERFACES.items():
        reconstructed = np.stack(
            [interface_transform(rpm, interface, limits) for rpm in raw_rpm]
        )
        reconstructed_wrench = np.stack(
            [
                motor_wrench(
                    rpm,
                    kf=sample_rows[0]["kf"],
                    km=sample_rows[0]["km"],
                    arm_length=sample_rows[0]["arm_length"],
                )
                for rpm in reconstructed
            ]
        )
        rpm_error = reconstructed - raw_rpm
        wrench_error = reconstructed_wrench - raw_wrench
        torque_gain = []
        torque_direction_change = []
        for axis in range(1, 4):
            raw_axis = raw_wrench[:, axis]
            reconstructed_axis = reconstructed_wrench[:, axis]
            denominator = float(np.dot(raw_axis, raw_axis))
            torque_gain.append(
                1.0
                if denominator <= 1e-24
                else float(np.dot(raw_axis, reconstructed_axis) / denominator)
            )
            significant = np.abs(raw_axis) >= max(
                1e-10,
                0.01 * float(np.max(np.abs(raw_axis))),
            )
            torque_direction_change.append(
                0.0
                if not np.any(significant)
                else float(
                    np.mean(
                        np.sign(raw_axis[significant])
                        != np.sign(reconstructed_axis[significant])
                    )
                )
            )

        roll_raw = raw_wrench[recovery_roll_samples, 1]
        roll_reconstructed = reconstructed_wrench[recovery_roll_samples, 1]
        roll_denominator = float(np.dot(roll_raw, roll_raw))
        summaries[interface_name] = {
            "sample_count": int(raw_rpm.shape[0]),
            "physical_saturation_sample_fraction": float(
                np.mean(np.any(physical_saturated, axis=1))
            ),
            "modified_action_fraction": float(
                np.mean(np.any(np.abs(rpm_error) > 1e-6, axis=1))
            ),
            "unsaturated_rpm_error_mean_abs": float(
                np.mean(np.abs(rpm_error[unsaturated_samples]))
            ),
            "unsaturated_rpm_error_max_abs": float(
                np.max(np.abs(rpm_error[unsaturated_samples]))
            ),
            "per_motor_rpm_error_mean_abs": np.mean(
                np.abs(rpm_error), axis=0
            ).tolist(),
            "per_motor_rpm_error_p95_abs": np.percentile(
                np.abs(rpm_error), 95, axis=0
            ).tolist(),
            "mean_abs_total_thrust_error": float(
                np.mean(np.abs(wrench_error[:, 0]))
            ),
            "relative_mean_abs_total_thrust_error": float(
                np.mean(np.abs(wrench_error[:, 0]))
                / max(1e-12, np.mean(np.abs(raw_wrench[:, 0])))
            ),
            "roll_torque_gain": torque_gain[0],
            "pitch_torque_gain": torque_gain[1],
            "yaw_torque_gain": torque_gain[2],
            "roll_direction_change_fraction": torque_direction_change[0],
            "pitch_direction_change_fraction": torque_direction_change[1],
            "yaw_direction_change_fraction": torque_direction_change[2],
            "recovery_roll_torque_gain": (
                1.0
                if roll_denominator <= 1e-24
                else float(
                    np.dot(roll_raw, roll_reconstructed) / roll_denominator
                )
            ),
            "recovery_roll_peak_retention": float(
                np.max(np.abs(roll_reconstructed))
                / max(1e-12, np.max(np.abs(roll_raw)))
            ),
        }
    return summaries


def summarize_closed_loop(results):
    summaries = {}
    for interface_name in INTERFACES:
        selected = [result for result in results if result["interface"] == interface_name]
        summaries[interface_name] = {
            "episodes": len(selected),
            "full_horizon_rate": float(np.mean([x["full_horizon"] for x in selected])),
            "instability_rate": float(np.mean([x["unstable"] for x in selected])),
            "attitude_bound_rate": float(
                np.mean([x["done_reason"] == "attitude_bound" for x in selected])
            ),
            "xy_bound_rate": float(
                np.mean([x["done_reason"] == "xy_bound" for x in selected])
            ),
            "height_bound_rate": float(
                np.mean([x["done_reason"] == "height_bound" for x in selected])
            ),
            "mean_max_position_error": float(
                np.mean([x["max_position_error"] for x in selected])
            ),
            "worst_max_abs_roll": float(max(x["max_abs_roll"] for x in selected)),
            "worst_max_abs_pitch": float(max(x["max_abs_pitch"] for x in selected)),
            "worst_max_abs_yaw": float(max(x["max_abs_yaw"] for x in selected)),
            "worst_max_angular_velocity": float(
                max(x["max_angular_velocity"] for x in selected)
            ),
            "mean_action_modified_fraction": float(
                np.mean([x["action_modified_fraction"] for x in selected])
            ),
            "physical_saturation_fraction": float(
                np.mean([x["physical_saturation_fraction"] for x in selected])
            ),
        }
    return summaries


def evaluate_gate(closed_loop, offline, episodes):
    interface_gates = {}
    raw_lookup = {
        (episode["case"], episode["seed"]): episode
        for episode in episodes
        if episode["interface"] == "T1_raw_dslpid_rpm"
    }
    for interface_name in ("T3_asymmetric_rpm", "T4_asymmetric_thrust"):
        closed = closed_loop[interface_name]
        numerical = offline[interface_name]
        selected = [
            episode for episode in episodes if episode["interface"] == interface_name
        ]
        required_disturbances = [
            episode
            for episode in selected
            if episode["case"]
            not in {"hover_zero", "straight_reference", "gentle_curve_reference"}
        ]
        raw_match = []
        for episode in selected:
            raw = raw_lookup[(episode["case"], episode["seed"])]
            raw_match.append(
                episode["done_reason"] == raw["done_reason"]
                and episode["steps"] == raw["steps"]
                and np.isclose(
                    episode["max_position_error"],
                    raw["max_position_error"],
                    atol=1e-9,
                    rtol=1e-9,
                )
                and (
                    episode["recovery_time"] == raw["recovery_time"]
                    or (
                        episode["recovery_time"] is not None
                        and raw["recovery_time"] is not None
                        and np.isclose(
                            episode["recovery_time"],
                            raw["recovery_time"],
                            atol=1.0 / 120.0,
                        )
                    )
                )
            )
        checks = {
            "all_disturbances_reach_full_horizon": closed["full_horizon_rate"] == 1.0,
            "no_instability_bounds": closed["instability_rate"] == 0.0,
            "all_required_disturbances_recover": all(
                episode["recovery_time"] is not None
                for episode in required_disturbances
            ),
            "closed_loop_matches_raw_teacher": all(raw_match),
            "unsaturated_roundtrip_max_rpm_error_le_1e-6": (
                numerical["unsaturated_rpm_error_max_abs"] <= 1e-6
            ),
            "no_nonphysical_action_projection": (
                numerical["modified_action_fraction"]
                <= numerical["physical_saturation_sample_fraction"] + 1e-12
            ),
            "total_thrust_relative_error_le_1e-8": (
                numerical["relative_mean_abs_total_thrust_error"] <= 1e-8
            ),
            "roll_torque_gain_preserved": (
                0.98 <= numerical["roll_torque_gain"] <= 1.02
            ),
            "pitch_torque_gain_preserved": (
                0.98 <= numerical["pitch_torque_gain"] <= 1.02
            ),
            "yaw_torque_gain_preserved": (
                0.98 <= numerical["yaw_torque_gain"] <= 1.02
            ),
            "recovery_roll_torque_preserved": (
                0.98 <= numerical["recovery_roll_torque_gain"] <= 1.02
            ),
        }
        interface_gates[interface_name] = {
            "checks": checks,
            "passed": bool(all(checks.values())),
        }
    return {
        "interfaces": interface_gates,
        "motor_action_interface_gate_passed": bool(
            any(item["passed"] for item in interface_gates.values())
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="T1-T4 teacher feasibility and motor-action interface gate."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--episode-len-sec", type=float, default=12.0)
    parser.add_argument(
        "--output",
        type=str,
        default="runs/motor_action_interface_diagnostics/t1_t4_full.json",
    )
    args = parser.parse_args()

    cases = build_cases()
    results = []
    offline_samples = []
    total = len(INTERFACES) * len(cases) * len(args.seeds)
    completed = 0
    for interface_name, interface in INTERFACES.items():
        for case in cases:
            for seed in args.seeds:
                result, samples = episode(
                    interface_name,
                    interface,
                    case,
                    seed,
                    episode_len_sec=float(args.episode_len_sec),
                    collect_samples=(interface_name == "T1_raw_dslpid_rpm"),
                )
                results.append(result)
                offline_samples.extend(samples)
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(
                        f"[diagnostic] completed {completed}/{total}",
                        file=sys.stderr,
                        flush=True,
                    )

    offline_summary = summarize_offline_samples(offline_samples)
    closed_loop_summary = summarize_closed_loop(results)
    gate = evaluate_gate(closed_loop_summary, offline_summary, results)
    rendered = {
        "configuration": {
            "seeds": args.seeds,
            "episode_len_sec": float(args.episode_len_sec),
            "control_frequency_hz": 120,
            "interface_definitions": INTERFACES,
            "case_count": len(cases),
            "episode_count": len(results),
            "recovery_definition": (
                "position_error<=0.05m, max_abs_rpy<=0.05rad, "
                "velocity_error<=0.15m/s and angular_speed<=0.15rad/s "
                "for 30 consecutive control steps"
            ),
        },
        "teacher_lifecycle": {
            "controller_class": "gym_pybullet_drones.control.DSLPIDControl",
            "reset_called_once_per_episode": True,
        },
        "offline_same_state_action_analysis": offline_summary,
        "closed_loop_summary": closed_loop_summary,
        "gate": gate,
        "episodes": results,
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(rendered, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
