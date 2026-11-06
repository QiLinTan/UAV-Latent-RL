from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from scipy.spatial.transform import Rotation

from utils.gym_pybullet_compat import ensure_gym_pybullet_envs_compat

ensure_gym_pybullet_envs_compat()

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from data.bc_metrics import rpm_to_wrench
from models.motor_action_codec import (
    ASYMMETRIC_RPM,
    MotorActionCodec,
    MotorPhysicalLimits,
)
from scripts.bc_scenarios import build_scenarios, make_env


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Hold every observable teacher input fixed while varying only "
            "DSLPIDControl's hidden history state."
        )
    )
    parser.add_argument(
        "--output",
        default=(
            "runs/behavior_cloning/asymmetric_rpm_v2_plain_mlp_b4096_"
            "critical_gate_v2/teacher_hidden_state_aliasing.json"
        ),
    )
    args = parser.parse_args()

    scenario = next(
        item for item in build_scenarios() if item.name == "nominal_hover"
    )
    env = make_env(scenario, 1.0)
    parameters = {
        "min_rpm": 0.0,
        "hover_rpm": float(env.HOVER_RPM),
        "max_rpm": float(env.MAX_RPM),
        "kf": float(env.KF),
        "km": float(env.KM),
        "arm_length": float(env.L),
    }
    codec = MotorActionCodec(
        MotorPhysicalLimits(
            min_rpm=0.0,
            hover_rpm=float(env.HOVER_RPM),
            max_rpm=float(env.MAX_RPM),
            kf=float(env.KF),
        ),
        mode=ASYMMETRIC_RPM,
    )
    dt = float(env.CTRL_TIMESTEP)
    current_rpy = np.array([0.05, -0.04, 0.03], dtype=np.float64)
    current_angular_velocity = np.array([0.12, -0.08, 0.05], dtype=np.float64)
    fixed_inputs = {
        "control_timestep": dt,
        "cur_pos": np.array([0.08, -0.06, 1.08], dtype=np.float64),
        "cur_quat": Rotation.from_euler("XYZ", current_rpy).as_quat(),
        "cur_vel": np.array([0.10, -0.08, 0.04], dtype=np.float64),
        "cur_ang_vel": current_angular_velocity,
        "target_pos": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "target_vel": np.zeros(3, dtype=np.float64),
    }
    cases = [
        {
            "name": "reset_history",
            "last_rpy": np.zeros(3),
            "integral_pos_e": np.zeros(3),
            "integral_rpy_e": np.zeros(3),
        },
        {
            "name": "rate_consistent_history",
            "last_rpy": current_rpy - current_angular_velocity * dt,
            "integral_pos_e": np.zeros(3),
            "integral_rpy_e": np.zeros(3),
        },
        {
            "name": "same_rate_with_accumulated_integrals",
            "last_rpy": current_rpy - current_angular_velocity * dt,
            "integral_pos_e": np.array([0.20, -0.15, 0.03]),
            "integral_rpy_e": np.array([0.0, 0.0, 0.02]),
        },
        {
            "name": "opposite_previous_rpy_history",
            "last_rpy": current_rpy + current_angular_velocity * dt,
            "integral_pos_e": np.array([-0.20, 0.15, -0.03]),
            "integral_rpy_e": np.array([0.0, 0.0, -0.02]),
        },
    ]
    results = []
    for case in cases:
        teacher = DSLPIDControl(drone_model=env.DRONE_MODEL)
        teacher.reset()
        teacher.last_rpy = case["last_rpy"].copy()
        teacher.integral_pos_e = case["integral_pos_e"].copy()
        teacher.integral_rpy_e = case["integral_rpy_e"].copy()
        rpm, _, _ = teacher.computeControl(**fixed_inputs)
        action = codec.rpm_to_normalized_action(rpm)
        wrench = rpm_to_wrench(rpm.reshape(1, -1), parameters)[0]
        results.append(
            {
                "name": case["name"],
                "hidden_state_before_call": {
                    "last_rpy": case["last_rpy"].tolist(),
                    "integral_pos_e": case["integral_pos_e"].tolist(),
                    "integral_rpy_e": case["integral_rpy_e"].tolist(),
                },
                "raw_teacher_rpm": rpm.tolist(),
                "encoded_teacher_action": action.tolist(),
                "teacher_wrench": {
                    "collective_thrust": float(wrench[0]),
                    "roll_torque": float(wrench[1]),
                    "pitch_torque": float(wrench[2]),
                    "yaw_torque": float(wrench[3]),
                },
            }
        )

    actions = np.asarray(
        [item["encoded_teacher_action"] for item in results],
        dtype=np.float64,
    )
    rpms = np.asarray(
        [item["raw_teacher_rpm"] for item in results],
        dtype=np.float64,
    )
    report = {
        "diagnostic": "same_observable_input_different_teacher_hidden_state",
        "all_computeControl_arguments_identical": True,
        "cur_ang_vel_is_present_in_actor_context_but_unused_by_DSLPIDControl": True,
        "teacher_output_depends_on_last_rpy": True,
        "teacher_output_depends_on_integral_pos_e": True,
        "teacher_output_depends_on_integral_rpy_e_yaw_only": True,
        "fixed_inputs": {
            name: np.asarray(value).tolist()
            if name != "control_timestep"
            else float(value)
            for name, value in fixed_inputs.items()
        },
        "cases": results,
        "cross_history_max_action_range": np.ptp(actions, axis=0).tolist(),
        "cross_history_max_rpm_range": np.ptp(rpms, axis=0).tolist(),
        "max_any_motor_action_range": float(np.max(np.ptp(actions, axis=0))),
        "max_any_motor_rpm_range": float(np.max(np.ptp(rpms, axis=0))),
        "conclusion": (
            "The present 30-D Actor context does not uniquely determine the "
            "DSLPID teacher label because last_rpy and integral states are omitted."
        ),
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}.")
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    env.close()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
