"""Quantify Stage2-v1 teacher-label ambiguity without training a network."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pybullet as p
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.gym_pybullet_compat import ensure_gym_pybullet_envs_compat

ensure_gym_pybullet_envs_compat()

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

from algos.td3.td3_reference_tracking import STAGE2_V1_CONTEXT_ID
from models.canonical_dslpid_teacher import (
    CANONICAL_DSLPID_WORLD_RATE_ID,
    CanonicalDSLPIDTeacher,
    WORLD_ANGULAR_VELOCITY,
    previous_rpy_from_world_angular_velocity,
)
from models.motor_action_codec import (
    ASYMMETRIC_RPM,
    MotorActionCodec,
    MotorPhysicalLimits,
)


CLOSE_CONTEXT_RADIUS = 0.05


def nearest_cross_episode_pairs(
    contexts: np.ndarray,
    episode_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find each sample's nearest context from a different episode."""

    contexts = np.asarray(contexts, dtype=np.float64)
    episode_ids = np.asarray(episode_ids).reshape(-1)
    if contexts.ndim != 2 or contexts.shape[0] != episode_ids.size:
        raise ValueError("contexts and episode_ids have incompatible shapes.")
    unique_episodes = np.unique(episode_ids)
    if unique_episodes.size < 2:
        raise ValueError("Cross-episode ambiguity requires at least two episodes.")

    source_indices = []
    neighbor_indices = []
    pair_distances = []
    all_indices = np.arange(contexts.shape[0])
    for episode_id in unique_episodes:
        source = all_indices[episode_ids == episode_id]
        candidates = all_indices[episode_ids != episode_id]
        tree = cKDTree(contexts[candidates])
        distances, local_indices = tree.query(contexts[source], k=1)
        source_indices.append(source)
        neighbor_indices.append(candidates[np.asarray(local_indices, dtype=np.int64)])
        pair_distances.append(np.asarray(distances, dtype=np.float64))
    return (
        np.concatenate(source_indices),
        np.concatenate(neighbor_indices),
        np.concatenate(pair_distances),
    )


def _percentiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "max": float(np.max(values)),
    }


def local_ambiguity_metrics(
    contexts: np.ndarray,
    actions: np.ndarray,
    wrenches: np.ndarray,
    episode_ids: np.ndarray,
    hidden_states: np.ndarray,
    *,
    close_radius: float = CLOSE_CONTEXT_RADIUS,
    torque_thresholds: np.ndarray | None = None,
    torque_sign_is_gate: bool = True,
) -> dict:
    source, neighbor, distances = nearest_cross_episode_pairs(contexts, episode_ids)
    action_delta = np.asarray(actions[source] - actions[neighbor], dtype=np.float64)
    action_linf = np.max(np.abs(action_delta), axis=1)
    close = distances <= float(close_radius)
    close_count = int(np.sum(close))
    if close_count:
        close_delta = action_delta[close]
        close_linf = action_linf[close]
        pair_action_rmse = float(np.sqrt(np.mean(np.square(close_delta))))
        local_label_dispersion_rmse = pair_action_rmse / np.sqrt(2.0)
    else:
        close_linf = np.asarray([np.inf])
        local_label_dispersion_rmse = float("inf")

    teacher_action_rms = float(np.sqrt(np.mean(np.square(actions))))
    dispersion_ratio = local_label_dispersion_rmse / max(teacher_action_rms, 1e-12)
    torque_values = np.asarray(wrenches[:, 1:4], dtype=np.float64)
    if torque_thresholds is None:
        torque_thresholds = np.maximum(
            np.percentile(np.abs(torque_values), 90.0, axis=0),
            1e-12,
        )
    torque_thresholds = np.asarray(torque_thresholds, dtype=np.float64).reshape(3)
    first_torque = torque_values[source]
    second_torque = torque_values[neighbor]
    salient = np.logical_and(
        np.abs(first_torque) >= torque_thresholds,
        np.abs(second_torque) >= torque_thresholds,
    )
    sign_conflict = np.logical_and(
        salient,
        np.sign(first_torque) != np.sign(second_torque),
    )
    salient_count = int(np.sum(salient))
    sign_conflict_rate = float(np.sum(sign_conflict) / max(1, salient_count))

    hidden_delta = np.linalg.norm(
        np.asarray(hidden_states[source] - hidden_states[neighbor], dtype=np.float64),
        axis=1,
    )
    if np.std(hidden_delta) > 0.0 and np.std(action_linf) > 0.0:
        hidden_action_correlation = float(np.corrcoef(hidden_delta, action_linf)[0, 1])
    else:
        hidden_action_correlation = 0.0
    minimum_close_count = max(100, int(np.ceil(0.01 * contexts.shape[0])))
    checks = {
        "sufficient_close_pairs": close_count >= minimum_close_count,
        "close_pair_p95_action_linf": float(np.percentile(close_linf, 95.0)) <= 0.05,
        "local_dispersion_ratio": bool(dispersion_ratio <= 0.20),
    }
    if torque_sign_is_gate:
        checks["salient_torque_sign_conflict"] = sign_conflict_rate <= 0.02
    return {
        "sample_count": int(contexts.shape[0]),
        "close_context_radius": float(close_radius),
        "cross_episode_context_distance": _percentiles(distances),
        "close_pair_count": close_count,
        "close_pair_fraction": float(close_count / contexts.shape[0]),
        "minimum_close_pair_count": minimum_close_count,
        "close_pair_action_linf": _percentiles(close_linf),
        "teacher_action_rms": teacher_action_rms,
        "local_label_dispersion_rmse": float(local_label_dispersion_rmse),
        "local_dispersion_to_teacher_rms_ratio": float(dispersion_ratio),
        "salient_torque_thresholds": torque_thresholds.tolist(),
        "salient_torque_comparison_count": salient_count,
        "salient_torque_sign_conflict_rate": sign_conflict_rate,
        "hidden_state_delta_to_action_linf_correlation": hidden_action_correlation,
        "gate_checks": checks,
        "gate_passed": bool(all(checks.values())),
    }


def canonical_teacher_reproduction_metrics(
    samples: dict[str, np.ndarray],
    physical_parameters: dict,
    *,
    sample_count: int = 256,
) -> dict:
    """Independently regenerate sampled canonical labels from stored observables."""

    if "state_quaternion" not in samples:
        raise ValueError("Canonical dataset must store state_quaternion.")
    total = int(samples["context"].shape[0])
    selected = np.unique(
        np.linspace(0, total - 1, min(int(sample_count), total), dtype=np.int64)
    )
    codec = MotorActionCodec(
        MotorPhysicalLimits(
            min_rpm=float(physical_parameters["min_rpm"]),
            hover_rpm=float(physical_parameters["hover_rpm"]),
            max_rpm=float(physical_parameters["max_rpm"]),
            kf=float(physical_parameters["kf"]),
        ),
        mode=ASYMMETRIC_RPM,
    )
    teacher = CanonicalDSLPIDTeacher(
        DroneModel.CF2X,
        rate_mode=WORLD_ANGULAR_VELOCITY,
    )
    rpm_errors = []
    action_errors = []
    hidden_errors = []
    repeated_rpm_errors = []
    for index in selected:
        state = np.asarray(samples["state"][index], dtype=np.float64)
        quaternion = np.asarray(
            samples["state_quaternion"][index],
            dtype=np.float64,
        )
        reference = np.asarray(samples["current_reference"][index], dtype=np.float64)
        inputs = {
            "control_timestep": 1.0 / 120.0,
            "cur_pos": state[:3],
            "cur_quat": quaternion,
            "cur_vel": state[6:9],
            "cur_ang_vel": state[9:12],
            "target_pos": reference[:3],
            "target_vel": reference[3:],
        }
        regenerated_rpm = np.asarray(teacher.computeControl(**inputs)[0], dtype=np.float64)
        hidden_before = teacher.last_hidden_before
        repeated_rpm = np.asarray(teacher.computeControl(**inputs)[0], dtype=np.float64)
        stored_rpm = np.asarray(samples["raw_teacher_rpm"][index], dtype=np.float64)
        stored_action = np.asarray(
            samples["encoded_teacher_action"][index],
            dtype=np.float64,
        )
        regenerated_action = np.asarray(
            codec.rpm_to_normalized_action(regenerated_rpm),
            dtype=np.float64,
        )
        rpm_errors.append(np.max(np.abs(regenerated_rpm - stored_rpm)))
        action_errors.append(np.max(np.abs(regenerated_action - stored_action)))
        repeated_rpm_errors.append(np.max(np.abs(repeated_rpm - regenerated_rpm)))
        stored_hidden = np.concatenate(
            [
                samples["teacher_integral_pos_e_before"][index],
                samples["teacher_integral_rpy_e_before"][index],
                samples["teacher_last_rpy_before"][index],
                samples["teacher_last_rpy_e_before"][index],
                np.asarray([samples["teacher_control_counter_before"][index]]),
            ]
        ).astype(np.float64)
        regenerated_hidden = np.concatenate(
            [
                hidden_before["integral_pos_e"],
                hidden_before["integral_rpy_e"],
                hidden_before["last_rpy"],
                hidden_before["last_rpy_e"],
                np.asarray([hidden_before["control_counter"]]),
            ]
        ).astype(np.float64)
        hidden_errors.append(np.max(np.abs(regenerated_hidden - stored_hidden)))

    rpm_errors = np.asarray(rpm_errors, dtype=np.float64)
    action_errors = np.asarray(action_errors, dtype=np.float64)
    hidden_errors = np.asarray(hidden_errors, dtype=np.float64)
    repeated_rpm_errors = np.asarray(repeated_rpm_errors, dtype=np.float64)
    checks = {
        "stored_rpm_reproduced": float(np.max(rpm_errors)) <= 0.02,
        "stored_action_reproduced": float(np.max(action_errors)) <= 1e-5,
        "stored_hidden_input_reproduced": float(np.max(hidden_errors)) <= 1e-5,
        "repeated_call_exact": float(np.max(repeated_rpm_errors)) == 0.0,
    }
    return {
        "sample_count": int(selected.size),
        "rpm_linf_error": _percentiles(rpm_errors),
        "action_linf_error": _percentiles(action_errors),
        "hidden_input_linf_error": _percentiles(hidden_errors),
        "repeated_call_rpm_linf_error": _percentiles(repeated_rpm_errors),
        "gate_checks": checks,
        "gate_passed": bool(all(checks.values())),
    }


def physical_torque_salience_thresholds(
    physical_parameters: dict,
    wrenches: np.ndarray,
    *,
    capacity_fraction: float = 0.001,
) -> dict:
    """Combine a dataset percentile with a non-zero actuator-capacity floor."""

    max_rpm = float(physical_parameters["max_rpm"])
    kf = float(physical_parameters["kf"])
    km = float(physical_parameters["km"])
    arm_length = float(physical_parameters["arm_length"])
    roll_pitch_capacity = (
        2.0 * kf * max_rpm**2 * arm_length / np.sqrt(2.0)
    )
    yaw_capacity = 2.0 * km * max_rpm**2
    capacities = np.asarray(
        [roll_pitch_capacity, roll_pitch_capacity, yaw_capacity],
        dtype=np.float64,
    )
    percentile_thresholds = np.percentile(
        np.abs(np.asarray(wrenches, dtype=np.float64)[:, 1:4]),
        90.0,
        axis=0,
    )
    physical_floor = float(capacity_fraction) * capacities
    thresholds = np.maximum(percentile_thresholds, physical_floor)
    return {
        "capacity_fraction": float(capacity_fraction),
        "axis_capacities": capacities.tolist(),
        "dataset_p90": percentile_thresholds.tolist(),
        "physical_floor": physical_floor.tolist(),
        "thresholds": thresholds.tolist(),
    }
def _set_teacher_hidden_state(controller: DSLPIDControl, state: dict) -> None:
    controller.integral_pos_e = np.asarray(state["integral_pos_e"], dtype=np.float64).copy()
    controller.integral_rpy_e = np.asarray(state["integral_rpy_e"], dtype=np.float64).copy()
    controller.last_rpy = np.asarray(state["last_rpy"], dtype=np.float64).copy()
    controller.last_rpy_e = np.asarray(state["last_rpy_e"], dtype=np.float64).copy()
    controller.control_counter = int(state["control_counter"])


def counterfactual_hidden_state_metrics(
    samples: dict[str, np.ndarray],
    physical_parameters: dict,
    *,
    sample_count: int = 48,
) -> dict:
    total = int(samples["context"].shape[0])
    selected = np.unique(
        np.linspace(0, total - 1, min(int(sample_count), total), dtype=np.int64)
    )
    codec = MotorActionCodec(
        MotorPhysicalLimits(
            min_rpm=float(physical_parameters["min_rpm"]),
            hover_rpm=float(physical_parameters["hover_rpm"]),
            max_rpm=float(physical_parameters["max_rpm"]),
            kf=float(physical_parameters["kf"]),
        ),
        mode=ASYMMETRIC_RPM,
    )
    action_ranges = []
    rpm_ranges = []
    observed_label_reproduction_errors = []
    effect_names = (
        "zero_position_integral",
        "zero_attitude_integral",
        "zero_rate_history_only",
        "canonical_zero_rate",
        "canonical_world_angular_velocity",
        "mirrored_all_history",
    )
    action_effects = {name: [] for name in effect_names}
    cases_per_sample = 1 + len(effect_names)
    for index in selected:
        state = np.asarray(samples["state"][index], dtype=np.float64)
        reference = np.asarray(samples["current_reference"][index], dtype=np.float64)
        observed = {
            "integral_pos_e": samples["teacher_integral_pos_e_before"][index],
            "integral_rpy_e": samples["teacher_integral_rpy_e_before"][index],
            "last_rpy": samples["teacher_last_rpy_before"][index],
            "last_rpy_e": samples["teacher_last_rpy_e_before"][index],
            "control_counter": samples["teacher_control_counter_before"][index],
        }
        zero_position_integral = dict(
            observed,
            integral_pos_e=np.zeros(3),
        )
        zero_attitude_integral = dict(
            observed,
            integral_rpy_e=np.zeros(3),
        )
        zero_rate_history = dict(
            observed,
            last_rpy=state[3:6].copy(),
        )
        canonical_zero_rate = {
            "integral_pos_e": np.zeros(3),
            "integral_rpy_e": np.zeros(3),
            "last_rpy": state[3:6].copy(),
            "last_rpy_e": np.zeros(3),
            "control_counter": 0,
        }
        canonical_world_rate = dict(
            canonical_zero_rate,
            last_rpy=previous_rpy_from_world_angular_velocity(
                p.getQuaternionFromEuler(state[3:6]),
                state[9:12],
                1.0 / 120.0,
            ),
        )
        mirrored = {
            "integral_pos_e": -np.asarray(observed["integral_pos_e"]),
            "integral_rpy_e": -np.asarray(observed["integral_rpy_e"]),
            "last_rpy": 2.0 * state[3:6] - np.asarray(observed["last_rpy"]),
            "last_rpy_e": -np.asarray(observed["last_rpy_e"]),
            "control_counter": int(observed["control_counter"]),
        }
        cases = {
            "observed": observed,
            "zero_position_integral": zero_position_integral,
            "zero_attitude_integral": zero_attitude_integral,
            "zero_rate_history_only": zero_rate_history,
            "canonical_zero_rate": canonical_zero_rate,
            "canonical_world_angular_velocity": canonical_world_rate,
            "mirrored_all_history": mirrored,
        }
        rpms = {}
        actions = {}
        for name, hidden in cases.items():
            teacher = DSLPIDControl(drone_model=DroneModel.CF2X)
            teacher.reset()
            _set_teacher_hidden_state(teacher, hidden)
            rpm, _, _ = teacher.computeControl(
                control_timestep=1.0 / 120.0,
                cur_pos=state[:3],
                cur_quat=p.getQuaternionFromEuler(state[3:6]),
                cur_vel=state[6:9],
                cur_ang_vel=state[9:12],
                target_pos=reference[:3],
                target_vel=reference[3:],
            )
            rpms[name] = np.asarray(rpm, dtype=np.float64)
            actions[name] = np.asarray(
                codec.rpm_to_normalized_action(rpm),
                dtype=np.float64,
            )
        observed_label_reproduction_errors.append(
            np.max(
                np.abs(
                    rpms["observed"]
                    - np.asarray(samples["raw_teacher_rpm"][index], dtype=np.float64)
                )
            )
        )
        for name in effect_names:
            action_effects[name].append(
                np.max(np.abs(actions[name] - actions["observed"]))
            )
        action_matrix = np.stack(list(actions.values()))
        rpm_matrix = np.stack(list(rpms.values()))
        action_ranges.append(np.max(np.ptp(action_matrix, axis=0)))
        rpm_ranges.append(np.max(np.ptp(rpm_matrix, axis=0)))

    action_ranges = np.asarray(action_ranges, dtype=np.float64)
    rpm_ranges = np.asarray(rpm_ranges, dtype=np.float64)
    reproduction_errors = np.asarray(
        observed_label_reproduction_errors,
        dtype=np.float64,
    )
    component_effects = {
        name: _percentiles(np.asarray(values, dtype=np.float64))
        for name, values in action_effects.items()
    }
    checks = {
        "observed_label_reproduction": float(np.max(reproduction_errors)) <= 0.01,
        "p95_any_motor_action_range": float(np.percentile(action_ranges, 95.0)) <= 0.05,
        "max_any_motor_action_range": float(np.max(action_ranges)) <= 0.10,
    }
    return {
        "sample_count": int(selected.size),
        "cases_per_exact_context": cases_per_sample,
        "all_observable_inputs_identical_within_each_case_set": True,
        "varied_fields": [
            "teacher_integral_pos_e",
            "teacher_integral_rpy_e",
            "teacher_last_rpy",
            "teacher_last_rpy_e",
            "teacher_control_counter",
        ],
        "component_action_linf_effects": component_effects,
        "any_motor_action_range": _percentiles(action_ranges),
        "any_motor_rpm_range": _percentiles(rpm_ranges),
        "observed_label_reproduction_rpm_error": _percentiles(reproduction_errors),
        "gate_checks": checks,
        "gate_passed": bool(all(checks.values())),
    }


def analyze_dataset(dataset_dir: Path, *, counterfactual_samples: int = 48) -> dict:
    dataset_dir = Path(dataset_dir)
    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("context_contract_id") != STAGE2_V1_CONTEXT_ID:
        raise ValueError("Dataset does not declare the frozen Stage2-v1 context.")
    if metadata.get("motor_action_codec") != ASYMMETRIC_RPM:
        raise ValueError("Dataset does not use asymmetric_rpm.")
    teacher_contract_id = metadata.get("teacher_contract_id")
    canonical_teacher = teacher_contract_id == CANONICAL_DSLPID_WORLD_RATE_ID
    actual_hash = hashlib.sha256((dataset_dir / "samples.npz").read_bytes()).hexdigest()
    if actual_hash != metadata.get("samples_sha256"):
        raise ValueError("Stage2-v1 samples hash mismatch.")
    with np.load(dataset_dir / "samples.npz") as loaded:
        samples = {name: loaded[name] for name in loaded.files}

    hidden = np.concatenate(
        [
            samples["teacher_integral_pos_e_before"],
            samples["teacher_integral_rpy_e_before"],
            samples["teacher_last_rpy_before"],
            samples["teacher_last_rpy_e_before"],
            samples["teacher_control_counter_before"].reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float64)
    salience = physical_torque_salience_thresholds(
        metadata["physical_parameters"],
        samples["teacher_wrench"],
    )
    torque_thresholds = np.asarray(salience["thresholds"], dtype=np.float64)
    overall = local_ambiguity_metrics(
        samples["context"],
        samples["encoded_teacher_action"],
        samples["teacher_wrench"],
        samples["episode_id"],
        hidden,
        torque_thresholds=torque_thresholds,
        torque_sign_is_gate=not canonical_teacher,
    )
    by_task = {}
    task_mapping = {int(value): name for name, value in metadata["task_id_mapping"].items()}
    for task_id, task_name in sorted(task_mapping.items()):
        mask = samples["task_id"] == task_id
        by_task[task_name] = local_ambiguity_metrics(
            samples["context"][mask],
            samples["encoded_teacher_action"][mask],
            samples["teacher_wrench"][mask],
            samples["episode_id"][mask],
            hidden[mask],
            torque_thresholds=torque_thresholds,
            torque_sign_is_gate=not canonical_teacher,
        )
    counterfactual = counterfactual_hidden_state_metrics(
        samples,
        metadata["physical_parameters"],
        sample_count=counterfactual_samples,
    )
    canonical_reproduction = None
    if canonical_teacher:
        canonical_reproduction = canonical_teacher_reproduction_metrics(
            samples,
            metadata["physical_parameters"],
            sample_count=max(256, int(counterfactual_samples)),
        )
    local_passed = bool(
        overall["gate_passed"]
        and all(value["gate_passed"] for value in by_task.values())
    )
    teacher_uniqueness_passed = bool(
        canonical_reproduction["gate_passed"]
        if canonical_teacher
        else counterfactual["gate_passed"]
    )
    collection_gate_passed = bool(
        metadata.get("collection_gate_passed", metadata.get("pilot_gate_passed", False))
    )
    ambiguity_passed = bool(
        collection_gate_passed and local_passed and teacher_uniqueness_passed
    )
    return {
        "analysis": "stage2_v1_same_context_teacher_label_ambiguity",
        "dataset_dir": str(dataset_dir),
        "dataset_version": metadata["dataset_version"],
        "samples_sha256_verified": True,
        "context_contract_id": STAGE2_V1_CONTEXT_ID,
        "teacher_contract_id": teacher_contract_id,
        "network_training_performed": False,
        "method": {
            "local": (
                "nearest normalized 30D context from a different episode; "
                "close pairs use Euclidean radius 0.05"
            ),
            "counterfactual": (
                "hold all 30D-observable control inputs fixed and vary only "
                "DSLPID hidden history state"
            ),
        },
        "physical_torque_salience": salience,
        "local_ambiguity_overall": overall,
        "local_ambiguity_by_task": by_task,
        "counterfactual_hidden_state": counterfactual,
        "canonical_teacher_reproduction": canonical_reproduction,
        "gate_checks": {
            "collection_gate_passed": collection_gate_passed,
            "local_context_label_smoothness": local_passed,
            "selected_teacher_label_uniqueness": teacher_uniqueness_passed,
        },
        "ambiguity_gate_passed": ambiguity_passed,
        "training_allowed": ambiguity_passed,
        "recommended_next_step": (
            (
                "proceed_to_bc_warm_start"
                if metadata.get("profile") == "formal_bc_candidate"
                else "proceed_to_full_stage2_v1_collection"
            )
            if ambiguity_passed
            else "hold_full_collection_and_review_teacher_label_ambiguity"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default="data/behavior_cloning/stage2_v1_external_packet_pilot",
    )
    parser.add_argument(
        "--output",
        default="analysis_outputs/stage2_v1_label_ambiguity/pilot_report.json",
    )
    parser.add_argument("--counterfactual-samples", type=int, default=48)
    parser.add_argument(
        "--release-output",
        help="Optional dataset-local quality gate file used to release BC training.",
    )
    args = parser.parse_args()
    report = analyze_dataset(
        Path(args.dataset_dir),
        counterfactual_samples=args.counterfactual_samples,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}.")
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    output.write_text(rendered + "\n", encoding="utf-8")
    if args.release_output:
        release_output = Path(args.release_output)
        if release_output.exists():
            raise FileExistsError(f"Refusing to overwrite {release_output}.")
        release_output.parent.mkdir(parents=True, exist_ok=True)
        release = {
            "dataset_dir": report["dataset_dir"],
            "dataset_version": report["dataset_version"],
            "samples_sha256_verified": report["samples_sha256_verified"],
            "context_contract_id": report["context_contract_id"],
            "teacher_contract_id": report["teacher_contract_id"],
            "ambiguity_gate_passed": report["ambiguity_gate_passed"],
            "training_allowed": report["training_allowed"],
            "gate_checks": report["gate_checks"],
            "analysis_report": str(output),
        }
        release_output.write_text(
            json.dumps(release, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(rendered)


if __name__ == "__main__":
    main()
