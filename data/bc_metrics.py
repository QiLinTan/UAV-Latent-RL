from __future__ import annotations

import numpy as np


def decode_asymmetric_rpm(actions, physical_parameters):
    actions = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
    minimum = float(physical_parameters["min_rpm"])
    hover = float(physical_parameters["hover_rpm"])
    maximum = float(physical_parameters["max_rpm"])
    return np.where(
        actions >= 0.0,
        hover + actions * (maximum - hover),
        hover + actions * (hover - minimum),
    )


def rpm_to_wrench(rpm, physical_parameters):
    rpm = np.asarray(rpm, dtype=np.float64)
    kf = float(physical_parameters["kf"])
    km = float(physical_parameters["km"])
    arm = float(physical_parameters["arm_length"])
    forces = kf * np.square(rpm)
    squared = np.square(rpm)
    total = np.sum(forces, axis=1)
    roll = (
        forces[:, 0] + forces[:, 1] - forces[:, 2] - forces[:, 3]
    ) * arm / np.sqrt(2.0)
    pitch = (
        -forces[:, 0] + forces[:, 1] + forces[:, 2] - forces[:, 3]
    ) * arm / np.sqrt(2.0)
    yaw = (
        -squared[:, 0] + squared[:, 1] - squared[:, 2] + squared[:, 3]
    ) * km
    return np.stack([total, roll, pitch, yaw], axis=1)


def _channel_errors(prediction, target, channel_names):
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(
        target,
        dtype=np.float64,
    )
    absolute = np.abs(error)
    return {
        "mae": {
            name: float(np.mean(absolute[:, index]))
            for index, name in enumerate(channel_names)
        },
        "rmse": {
            name: float(np.sqrt(np.mean(np.square(error[:, index]))))
            for index, name in enumerate(channel_names)
        },
        "max_abs": {
            name: float(np.max(absolute[:, index]))
            for index, name in enumerate(channel_names)
        },
        "absolute_error_percentiles": {
            name: {
                str(percentile): float(
                    np.percentile(absolute[:, index], percentile)
                )
                for percentile in (50, 90, 95, 99, 99.9)
            }
            for index, name in enumerate(channel_names)
        },
    }


def metric_block(actor_action, teacher_action, physical_parameters):
    actor_action = np.asarray(actor_action, dtype=np.float64)
    teacher_action = np.asarray(teacher_action, dtype=np.float64)
    if actor_action.shape[0] == 0:
        return {"sample_count": 0}
    actor_rpm = decode_asymmetric_rpm(actor_action, physical_parameters)
    teacher_rpm = decode_asymmetric_rpm(teacher_action, physical_parameters)
    actor_wrench = rpm_to_wrench(actor_rpm, physical_parameters)
    teacher_wrench = rpm_to_wrench(teacher_rpm, physical_parameters)
    wrench_names = ("collective_thrust", "roll_torque", "pitch_torque", "yaw_torque")
    wrench_metrics = _channel_errors(actor_wrench, teacher_wrench, wrench_names)
    direction_agreement = {}
    for index, name in enumerate(wrench_names[1:], start=1):
        target = teacher_wrench[:, index]
        prediction = actor_wrench[:, index]
        threshold = max(1e-12, 0.01 * float(np.max(np.abs(target))))
        significant = np.abs(target) >= threshold
        direction_agreement[name] = (
            1.0
            if not np.any(significant)
            else float(
                np.mean(
                    np.sign(target[significant])
                    == np.sign(prediction[significant])
                )
            )
        )
    return {
        "sample_count": int(actor_action.shape[0]),
        "action": _channel_errors(
            actor_action,
            teacher_action,
            ("motor_0", "motor_1", "motor_2", "motor_3"),
        ),
        "rpm": _channel_errors(
            actor_rpm,
            teacher_rpm,
            ("motor_0", "motor_1", "motor_2", "motor_3"),
        ),
        "wrench": wrench_metrics,
        "torque_direction_agreement": direction_agreement,
    }


def evaluate_grouped_predictions(
    actor_action,
    teacher_action,
    sample_arrays,
    sample_indices,
    physical_parameters,
):
    actor_action = np.asarray(actor_action)
    teacher_action = np.asarray(teacher_action)
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    groups = {
        "overall": np.ones(sample_indices.shape[0], dtype=bool),
        "nominal": sample_arrays["sample_group_id"][sample_indices] == 0,
        "initial_recovery": np.logical_and(
            sample_arrays["sample_group_id"][sample_indices] == 1,
            sample_arrays["recovery_flag"][sample_indices],
        ),
        "impulse_recovery": np.logical_and(
            sample_arrays["sample_group_id"][sample_indices] == 2,
            sample_arrays["recovery_flag"][sample_indices],
        ),
        "large_roll_torque": sample_arrays["large_roll_torque_flag"][
            sample_indices
        ],
        "large_pitch_torque": sample_arrays["large_pitch_torque_flag"][
            sample_indices
        ],
        "large_yaw_torque": sample_arrays["large_yaw_torque_flag"][
            sample_indices
        ],
    }
    return {
        name: metric_block(
            actor_action[mask],
            teacher_action[mask],
            physical_parameters,
        )
        for name, mask in groups.items()
    }
