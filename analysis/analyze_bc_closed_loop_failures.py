"""Diagnose BC planner failures without changing models, training, or sensors."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pybullet as p
import torch
from scipy.spatial.distance import cdist

from async_control import AsyncBuffer, AsyncScheduler, AsyncSchedulerConfig, LowLevelReferenceSelector
from envs.ForestAviary import CustomForestAviary
from envs.preprocess import preprocess_state
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from high_level_planner import (
    FUTURE_TIMES,
    ReferenceSequenceDecision,
    ReferenceSequencePacketGenerator,
    ReferenceSequencePlanner,
    extract_planner_input,
)
from scripts.evaluate_latent_async_pid import load_frozen_agent, make_motor_codec
from scripts.evaluate_native_latent_actor import clean_json, terminal_reason, write_json
from scripts.freeze_logical_async import file_record


SCENARIO_ORDER = ("clear", "single", "forest24")


def load_bc_dataset(root: Path) -> dict:
    metadata = json.loads((root / "metadata.json").read_text())
    with np.load(root / "samples.npz", allow_pickle=False) as samples:
        data = {name: samples[name].copy() for name in samples.files}
    with np.load(root / "splits.npz", allow_pickle=False) as splits:
        data["splits"] = {name: splits[name].copy() for name in splits.files}
    scenario_by_episode = {}
    start = 0
    for scenario in SCENARIO_ORDER:
        count = int(metadata["scenario_stats"][scenario]["successful_episodes"])
        for episode_id in range(start, start + count):
            scenario_by_episode[episode_id] = scenario
        start += count
    episodes = data["episode_id"].astype(np.int32)
    data["scenario"] = np.asarray([scenario_by_episode[int(ep)] for ep in episodes])
    data["metadata"] = metadata
    return data


def split_mask(data: dict, split: str) -> np.ndarray:
    return np.isin(data["episode_id"], data["splits"][split])


def standardize_inputs(planner_input: np.ndarray, train_inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_inputs.mean(axis=0)
    std = np.maximum(train_inputs.std(axis=0), 1e-3)
    return (planner_input - mean) / std, mean, std


def vector_heading_angle(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)) / denom, -1.0, 1.0))))


def summarize_bc_dataset(data: dict, dataset_root: Path) -> dict:
    metadata = data["metadata"]
    targets = data["reference_target"]
    first = targets[:, 0, :]
    last = targets[:, -1, :]
    train = split_mask(data, "train")
    summary = {
        "dataset": file_record(dataset_root / "samples.npz"),
        "metadata": file_record(dataset_root / "metadata.json"),
        "teacher_controller": metadata.get("teacher_controller"),
        "teacher_checkpoint": metadata.get("teacher_checkpoint"),
        "sample_count": int(metadata["sample_count"]),
        "episode_count": int(metadata["episode_count"]),
        "split_episode_counts": metadata["split_episode_counts"],
        "scenario_stats": metadata["scenario_stats"],
        "planner_input_dim": int(metadata["planner_input_dim"]),
        "target_shape": list(targets.shape[1:]),
        "successful_episodes_only": bool(metadata.get("successful_episodes_only", False)),
        "first_reference_offset_norm_mean_m": float(np.linalg.norm(first[:, :3], axis=1).mean()),
        "last_reference_offset_norm_mean_m": float(np.linalg.norm(last[:, :3], axis=1).mean()),
        "first_reference_speed_mean_mps": float(np.linalg.norm(first[:, 3:], axis=1).mean()),
        "last_reference_speed_mean_mps": float(np.linalg.norm(last[:, 3:], axis=1).mean()),
        "train_input_std_min": float(np.std(data["planner_input"][train], axis=0).min()),
        "train_input_std_median": float(np.median(np.std(data["planner_input"][train], axis=0))),
    }
    per_scenario = {}
    for scenario in SCENARIO_ORDER:
        mask = data["scenario"] == scenario
        per_scenario[scenario] = {
            "samples": int(mask.sum()),
            "episodes": int(np.unique(data["episode_id"][mask]).size),
            "first_offset_norm_mean_m": float(np.linalg.norm(first[mask, :3], axis=1).mean()),
            "first_offset_y_abs_mean_m": float(np.abs(first[mask, 1]).mean()),
            "first_velocity_y_abs_mean_mps": float(np.abs(first[mask, 4]).mean()),
        }
    summary["per_scenario"] = per_scenario
    return summary


def dataset_neighbor_conflicts(data: dict, scaled_inputs: np.ndarray, train_indices: np.ndarray) -> dict:
    refs = train_indices
    query = np.arange(len(data["planner_input"]))
    distances = cdist(scaled_inputs[query], scaled_inputs[refs])
    same_episode = data["episode_id"][query, None] == data["episode_id"][refs][None, :]
    distances = np.where(same_episode, np.inf, distances)
    nearest = np.argsort(distances, axis=1)[:, :5]
    nn_indices = refs[nearest]
    nn_distances = np.take_along_axis(distances, nearest, axis=1)
    q_target = data["reference_target"][query]
    n_target = data["reference_target"][nn_indices]
    q_first = q_target[:, 0, :2]
    n_first = n_target[:, :, 0, :2]
    q_final = q_target[:, -1, :2]
    n_final = n_target[:, :, -1, :2]
    first_delta = np.linalg.norm(q_first[:, None, :] - n_first, axis=2)
    final_delta = np.linalg.norm(q_final[:, None, :] - n_final, axis=2)
    full_position_rmse = np.sqrt(np.mean((q_target[:, None, :, :3] - n_target[:, :, :, :3]) ** 2, axis=(2, 3)))
    full_velocity_rmse = np.sqrt(np.mean((q_target[:, None, :, 3:] - n_target[:, :, :, 3:]) ** 2, axis=(2, 3)))
    angles = np.asarray(
        [
            [vector_heading_angle(q_first[i], n_first[i, j]) for j in range(nn_indices.shape[1])]
            for i in range(len(query))
        ]
    )
    bins = []
    for radius in (5.0, 7.5, 10.0, 12.5):
        mask = np.isfinite(nn_distances) & (nn_distances <= radius)
        bins.append(
            {
                "standardized_radius": radius,
                "pairs": int(mask.sum()),
                "position_delta_gt_0p15_rate": float((first_delta[mask] > 0.15).mean()) if mask.any() else None,
                "final_position_delta_gt_0p25_rate": float((final_delta[mask] > 0.25).mean()) if mask.any() else None,
                "full_position_rmse_gt_0p10_rate": float((full_position_rmse[mask] > 0.10).mean()) if mask.any() else None,
                "full_velocity_rmse_gt_0p25_rate": float((full_velocity_rmse[mask] > 0.25).mean()) if mask.any() else None,
                "heading_angle_gt_60deg_rate": float((angles[mask] > 60.0).mean()) if mask.any() else None,
                "heading_angle_gt_90deg_rate": float((angles[mask] > 90.0).mean()) if mask.any() else None,
            }
        )
    closest_conflicts = []
    score = np.where((full_position_rmse > 0.10) | (final_delta > 0.25) | (angles > 60.0), nn_distances, np.inf)
    for flat in np.argsort(score, axis=None)[:20]:
        i, j = np.unravel_index(flat, score.shape)
        if not np.isfinite(score[i, j]):
            continue
        neighbor = int(nn_indices[i, j])
        closest_conflicts.append(
            {
                "query_row": int(i),
                "neighbor_row": neighbor,
                "distance": float(nn_distances[i, j]),
                "query_episode": int(data["episode_id"][i]),
                "neighbor_episode": int(data["episode_id"][neighbor]),
                "query_scenario": str(data["scenario"][i]),
                "neighbor_scenario": str(data["scenario"][neighbor]),
                "first_position_delta_m": float(first_delta[i, j]),
                "final_position_delta_m": float(final_delta[i, j]),
                "full_position_rmse_m": float(full_position_rmse[i, j]),
                "full_velocity_rmse_mps": float(full_velocity_rmse[i, j]),
                "first_heading_angle_deg": float(angles[i, j]),
            }
        )
    return {
        "nearest_distance_median": float(np.median(nn_distances[:, 0][np.isfinite(nn_distances[:, 0])])),
        "nearest_distance_p95": float(np.percentile(nn_distances[:, 0][np.isfinite(nn_distances[:, 0])], 95.0)),
        "radius_analysis": bins,
        "closest_conflicts": closest_conflicts,
        "significant_conflict_definition": "same standardized planner-input neighborhood, full position RMSE >0.10m, final offset delta >0.25m, or first-heading angle >60deg",
    }


def load_planner(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    planner = ReferenceSequencePlanner(
        input_dim=int(checkpoint.get("input_dim", 27)),
        num_points=int(checkpoint.get("num_points", 5)),
    ).to(device)
    planner.load_state_dict(checkpoint["model_state_dict"])
    planner.eval()
    return planner, checkpoint


class PlannerPolicy:
    def __init__(self, agent, planner):
        self.agent = agent
        self.planner = planner

    def decision(self, state, env):
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.agent.device).reshape(1, -1)
        with torch.no_grad():
            latent = self.agent.encoder(tensor)[0]
        latent_value = latent.cpu().numpy().astype(np.float32)
        planner_input = extract_planner_input(latent_value, state)
        prediction = self.planner.predict_sequence(planner_input)
        return (
            planner_input,
            prediction,
            ReferenceSequenceDecision(
                position_offsets=prediction[:, :3],
                velocities=prediction[:, 3:],
                relative_times=FUTURE_TIMES,
                current_velocity=np.asarray(env.vel[0], dtype=np.float32),
                latent=latent_value,
            ),
        )


def make_env(seed: int, episode_len_sec: float):
    with contextlib.redirect_stdout(io.StringIO()):
        env = CustomForestAviary(
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            pyb_freq=240,
            ctrl_freq=120,
            gui=False,
            curriculum=False,
            num_trees=24,
            route_blocking_tree=True,
            route_blocking_tree_count=1,
            episode_len_sec=float(episode_len_sec),
        )
    return env


def predicted_world_points(prediction: np.ndarray, position: np.ndarray) -> np.ndarray:
    return np.asarray(position, dtype=np.float64).reshape(1, 3) + np.asarray(prediction, dtype=np.float64)[:, :3]


def sequence_min_clearance(points: np.ndarray, tree_specs) -> float:
    points = np.asarray(points, dtype=np.float64)
    if not tree_specs:
        return float("inf")
    clearances = []
    for spec in tree_specs:
        xy = np.asarray(spec["xy"], dtype=np.float64)
        clearances.append(float(np.linalg.norm(points[:, :2] - xy[None, :], axis=1).min() - float(spec["radius"]) - 0.06))
    return float(min(clearances))


def run_student_episode(seed: int, *, checkpoint: str, planner_checkpoint: Path, episode_len_sec: float, device: torch.device):
    env = make_env(seed, episode_len_sec)
    try:
        observation, _ = env.reset(seed=seed)
        state = preprocess_state(observation.reshape(-1))
        agent = load_frozen_agent(checkpoint, int(state.size), device)
        planner, _ = load_planner(planner_checkpoint, device)
        policy = PlannerPolicy(agent, planner)
        codec = make_motor_codec(env)
        pid = DSLPIDControl(drone_model=env.DRONE_MODEL)
        pid.reset()
        buffer = AsyncBuffer()
        generator = ReferenceSequencePacketGenerator(max_speed=0.8, max_acceleration=2.0, max_vertical_speed=0.4)
        scheduler = AsyncScheduler(
            AsyncSchedulerConfig(high_frequency_hz=15.0, delay_min_seconds=0.0, delay_max_seconds=0.05, skip_probability=0.0, seed=seed),
            generator,
            buffer,
        )
        lower = LowLevelReferenceSelector(env.pos[0])
        decisions = []
        states = []
        reward_sum = 0.0
        path_length = 0.0
        previous_position = np.asarray(env.pos[0], dtype=np.float64).copy()
        min_clearance = float("inf")
        info = {}
        terminated = truncated = False
        step = 0
        while not (terminated or truncated):
            now = step * float(env.CTRL_TIMESTEP)
            state_snapshot = state.copy()
            cache = {}

            def factory(_t_generate):
                planner_input, prediction, decision = policy.decision(state_snapshot, env)
                cache["planner_input"] = planner_input
                cache["prediction"] = prediction
                cache["position"] = np.asarray(env.pos[0], dtype=np.float32).copy()
                cache["velocity"] = np.asarray(env.vel[0], dtype=np.float32).copy()
                cache["clearance"] = float(env._computeNearestTreeClearance(env.pos[0]))
                cache["goal_distance"] = float(np.linalg.norm(np.asarray(env.TARGET_POS) - np.asarray(env.pos[0])))
                return decision

            scheduler.advance(
                now,
                factory,
                fallback_position=lower.last_position,
                fallback_velocity=np.zeros(3, dtype=np.float32),
            )
            if cache:
                world_points = predicted_world_points(cache["prediction"], cache["position"])
                decisions.append(
                    {
                        "step": int(step),
                        "time": float(now),
                        "planner_input": cache["planner_input"],
                        "prediction": cache["prediction"],
                        "position": cache["position"],
                        "velocity": cache["velocity"],
                        "goal_distance": cache["goal_distance"],
                        "clearance": cache["clearance"],
                        "predicted_sequence_min_clearance": sequence_min_clearance(world_points, env._tree_specs),
                    }
                )
            lower.select(buffer, now)
            rpm, _, _ = pid.computeControl(
                control_timestep=float(env.CTRL_TIMESTEP),
                cur_pos=np.asarray(env.pos[0], dtype=np.float64),
                cur_quat=np.asarray(p.getQuaternionFromEuler(env.rpy[0]), dtype=np.float64),
                cur_vel=np.asarray(env.vel[0], dtype=np.float64),
                cur_ang_vel=np.asarray(env.ang_v[0], dtype=np.float64),
                target_pos=np.asarray(lower.last_position, dtype=np.float64),
                target_vel=np.asarray(lower.last_velocity, dtype=np.float64),
            )
            action = codec.rpm_to_normalized_action(rpm).astype(np.float32)
            observation, reward, terminated, truncated, info = env.step(action.reshape(1, -1))
            position = np.asarray(env.pos[0], dtype=np.float64)
            path_length += float(np.linalg.norm(position - previous_position))
            previous_position = position.copy()
            min_clearance = min(min_clearance, float(info.get("min_tree_clearance", float("inf"))))
            state = preprocess_state(observation.reshape(-1))
            states.append(
                {
                    "step": int(step),
                    "position": position.copy(),
                    "velocity": np.asarray(env.vel[0], dtype=np.float32).copy(),
                    "goal_distance": float(info.get("goal_distance", np.nan)),
                    "clearance": float(info.get("min_tree_clearance", np.nan)),
                }
            )
            reward_sum += float(reward)
            step += 1
        return {
            "seed": int(seed),
            "steps": int(step),
            "success": bool(info.get("success", False)),
            "collision": bool(info.get("collision", False)),
            "done_reason": terminal_reason(env, info, terminated, truncated),
            "reward": float(reward_sum),
            "final_distance": float(np.linalg.norm(np.asarray(env.TARGET_POS) - np.asarray(env.pos[0]))),
            "minimum_clearance": float(min_clearance),
            "path_length": float(path_length),
            "decisions": decisions,
            "states": states,
            "trees": [
                {"xy": np.asarray(spec["xy"], dtype=float).tolist(), "radius": float(spec["radius"])}
                for spec in env._tree_specs
            ],
        }
    finally:
        env.close()


def annotate_student(episodes: list[dict], data: dict, scaled_dataset: np.ndarray, input_mean: np.ndarray, input_std: np.ndarray, train_indices: np.ndarray, ood_threshold: float, close_threshold: float) -> list[dict]:
    dataset_inputs = scaled_dataset[train_indices]
    annotated = []
    for episode in episodes:
        rows = []
        if episode["decisions"]:
            student_inputs = np.asarray([d["planner_input"] for d in episode["decisions"]], dtype=np.float32)
            scaled = (student_inputs - input_mean) / input_std
            distances = cdist(scaled, dataset_inputs)
            order = np.argsort(distances, axis=1)[:, :8]
            nn = train_indices[order]
            nn_dist = np.take_along_axis(distances, order, axis=1)
            for i, decision in enumerate(episode["decisions"]):
                neighbors = nn[i]
                teacher_refs = data["reference_target"][neighbors]
                pred = np.asarray(decision["prediction"], dtype=np.float32)
                first_deltas = np.linalg.norm(teacher_refs[:, 0, :2] - pred[None, 0, :2], axis=1)
                heading_angles = np.asarray([vector_heading_angle(pred[0, :2], ref[0, :2]) for ref in teacher_refs])
                neighbor_angles = []
                for a in range(len(neighbors)):
                    for b in range(a + 1, len(neighbors)):
                        neighbor_angles.append(vector_heading_angle(teacher_refs[a, 0, :2], teacher_refs[b, 0, :2]))
                ambiguity = bool(np.any(nn_dist[i] <= ood_threshold) and (np.max(neighbor_angles or [0.0]) > 60.0))
                rows.append(
                    {
                        "step": int(decision["step"]),
                        "time": float(decision["time"]),
                        "position": np.asarray(decision["position"]).tolist(),
                        "goal_distance": float(decision["goal_distance"]),
                        "clearance": float(decision["clearance"]),
                        "predicted_sequence_min_clearance": float(decision["predicted_sequence_min_clearance"]),
                        "nearest_train_distance": float(nn_dist[i, 0]),
                        "nearest_train_episode": int(data["episode_id"][neighbors[0]]),
                        "nearest_train_scenario": str(data["scenario"][neighbors[0]]),
                        "pred_vs_nn_first_position_delta_m": float(first_deltas[0]),
                        "pred_vs_nn_first_heading_angle_deg": float(heading_angles[0]),
                        "neighbor_reference_max_heading_angle_deg": float(np.max(neighbor_angles or [0.0])),
                        "ambiguity_distance_threshold": float(ood_threshold),
                        "local_observation_ambiguity": ambiguity,
                        "is_ood": bool(nn_dist[i, 0] > ood_threshold),
                        "predicted_imminent_collision": bool(decision["predicted_sequence_min_clearance"] < 0.05),
                    }
                )
        annotated.append({**{k: episode[k] for k in ("seed", "steps", "success", "collision", "done_reason", "reward", "final_distance", "minimum_clearance", "path_length")}, "decisions": rows})
    return annotated


def first_failure_reason(episode: dict, *, ood_threshold: float, close_threshold: float) -> dict:
    decisions = episode["decisions"]
    if not decisions:
        return {"category": "other", "first_error": None, "evidence": "No high-level decisions recorded."}
    ood = next((d for d in decisions if d["is_ood"]), None)
    ambiguity = next((d for d in decisions if d["local_observation_ambiguity"]), None)
    pred_collision = next((d for d in decisions if d["predicted_imminent_collision"]), None)
    tracking_to_collision = next((d for d in decisions if d["clearance"] < 0.25), None)
    large_pred_error = next(
        (
            d
            for d in decisions
            if d["nearest_train_distance"] <= close_threshold
            and (
                d["pred_vs_nn_first_position_delta_m"] > 0.20
                or d["pred_vs_nn_first_heading_angle_deg"] > 60.0
            )
        ),
        None,
    )
    candidates = []
    for category, row in (
        ("distribution_shift", ood),
        ("observation_ambiguity", ambiguity),
        ("other", pred_collision),
        ("other", large_pred_error),
    ):
        if row is not None:
            candidates.append((row["step"], category, row))
    if candidates:
        _, category, row = sorted(candidates, key=lambda item: item[0])[0]
        return {
            "category": category,
            "first_error_step": int(row["step"]),
            "first_error_time_seconds": float(row["time"]),
            "evidence": row,
        }
    if tracking_to_collision is not None:
        return {
            "category": "horizon_limitation",
            "first_error_step": int(tracking_to_collision["step"]),
            "first_error_time_seconds": float(tracking_to_collision["time"]),
            "evidence": {
                **tracking_to_collision,
                "interpretation": "No early OOD/ambiguity/large one-step planner mismatch was detected before clearance became small; failure may require longer temporal context or horizon.",
            },
        }
    return {
        "category": "other",
        "first_error_step": int(decisions[-1]["step"]),
        "first_error_time_seconds": float(decisions[-1]["time"]),
        "evidence": decisions[-1],
    }


def summarize_closed_loop(episodes: list[dict]) -> dict:
    return {
        "episodes": len(episodes),
        "success_rate": float(np.mean([ep["success"] for ep in episodes])),
        "collision_rate": float(np.mean([ep["collision"] for ep in episodes])),
        "mean_final_distance": float(np.mean([ep["final_distance"] for ep in episodes])),
        "mean_minimum_clearance": float(np.mean([ep["minimum_clearance"] for ep in episodes])),
        "worst_minimum_clearance": float(np.min([ep["minimum_clearance"] for ep in episodes])),
        "mean_path_length": float(np.mean([ep["path_length"] for ep in episodes])),
        "done_reasons": {reason: int(sum(ep["done_reason"] == reason for ep in episodes)) for reason in sorted({ep["done_reason"] for ep in episodes})},
    }


def make_report(args):
    dataset_root = Path(args.dataset)
    data = load_bc_dataset(dataset_root)
    train = split_mask(data, "train")
    train_indices = np.flatnonzero(train)
    scaled_all, input_mean, input_std = standardize_inputs(data["planner_input"], data["planner_input"][train])
    nn_train = cdist(scaled_all[train_indices], scaled_all[train_indices])
    np.fill_diagonal(nn_train, np.inf)
    same_ep = data["episode_id"][train_indices, None] == data["episode_id"][train_indices][None, :]
    nn_train = np.where(same_ep, np.inf, nn_train)
    train_nn = np.min(nn_train, axis=1)
    finite_train_nn = train_nn[np.isfinite(train_nn)]
    ood_threshold = float(np.percentile(finite_train_nn, 95.0))
    close_threshold = float(np.percentile(finite_train_nn, 25.0))

    device = torch.device("cpu")
    episodes = [
        run_student_episode(
            int(args.seed) + idx,
            checkpoint=args.checkpoint,
            planner_checkpoint=Path(args.planner),
            episode_len_sec=args.episode_len_sec,
            device=device,
        )
        for idx in range(args.episodes)
    ]
    annotated = annotate_student(
        episodes,
        data,
        scaled_all,
        input_mean,
        input_std,
        train_indices,
        ood_threshold,
        close_threshold,
    )
    failure_items = []
    for episode in annotated:
        if not episode["success"]:
            failure_items.append(
                {
                    "seed": episode["seed"],
                    "done_reason": episode["done_reason"],
                    "final_distance": episode["final_distance"],
                    "minimum_clearance": episode["minimum_clearance"],
                    **first_failure_reason(episode, ood_threshold=ood_threshold, close_threshold=close_threshold),
                }
            )
    categories = {name: int(sum(item["category"] == name for item in failure_items)) for name in ("distribution_shift", "observation_ambiguity", "horizon_limitation", "other")}

    student_distances = [
        d["nearest_train_distance"]
        for episode in annotated
        for d in episode["decisions"]
    ]
    student_pred_delta = [
        d["pred_vs_nn_first_position_delta_m"]
        for episode in annotated
        for d in episode["decisions"]
    ]
    student_heading_delta = [
        d["pred_vs_nn_first_heading_angle_deg"]
        for episode in annotated
        for d in episode["decisions"]
    ]
    recommendation = "both_dagger_and_temporal_context"
    report = {
        "scope": "diagnostic_only_no_model_training_no_sensor_changes",
        "dataset_summary": summarize_bc_dataset(data, dataset_root),
        "closed_loop_24tree_summary": summarize_closed_loop(annotated),
        "nearest_neighbor_calibration": {
            "train_train_nearest_distance_median": float(np.median(finite_train_nn)),
            "train_train_nearest_distance_p95": ood_threshold,
            "student_to_train_distance_median": float(np.median(student_distances)),
            "student_to_train_distance_p95": float(np.percentile(student_distances, 95.0)),
            "student_to_train_distance_max": float(np.max(student_distances)),
            "student_ood_fraction": float(np.mean(np.asarray(student_distances) > ood_threshold)),
        },
        "student_vs_nearest_teacher_reference": {
            "first_position_delta_median_m": float(np.median(student_pred_delta)),
            "first_position_delta_p95_m": float(np.percentile(student_pred_delta, 95.0)),
            "first_heading_angle_median_deg": float(np.median(student_heading_delta)),
            "first_heading_angle_p95_deg": float(np.percentile(student_heading_delta, 95.0)),
        },
        "bc_input_teacher_reference_conflicts": dataset_neighbor_conflicts(data, scaled_all, train_indices),
        "failure_episodes": failure_items,
        "failure_category_counts": categories,
        "recommendation": recommendation,
        "recommendation_evidence": [
            "DAgger is indicated when student closed-loop states leave the BC train manifold.",
            "Temporal context is indicated when similar 27D planner inputs map to materially different teacher references or when failures occur after locally plausible short-horizon decisions.",
            "Use this result as triage only: labels come from a privileged oracle A*/DSLPID teacher and the analysis is not a final navigation benchmark.",
        ],
        "inputs": {
            "dataset": str(dataset_root),
            "checkpoint": args.checkpoint,
            "planner": args.planner,
            "episodes": args.episodes,
            "seed": args.seed,
            "episode_len_sec": args.episode_len_sec,
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/reference_planner/latent_affordance_margin_v2")
    parser.add_argument("--checkpoint", default="checkpoints/latent_only_affordance_dim16/model_final")
    parser.add_argument("--planner", default="checkpoints/reference_planner/latent_affordance_margin_v2/planner_final.pt")
    parser.add_argument("--output", default="analysis_outputs/bc_failure_diagnostics/latent_affordance_margin_v2_forest24.json")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--episode-len-sec", type=float, default=12.0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = make_report(args)
    write_json(args.output, clean_json(report))
    print(json.dumps({
        "output": args.output,
        "closed_loop_success_rate": report["closed_loop_24tree_summary"]["success_rate"],
        "closed_loop_collision_rate": report["closed_loop_24tree_summary"]["collision_rate"],
        "student_ood_fraction": report["nearest_neighbor_calibration"]["student_ood_fraction"],
        "failure_category_counts": report["failure_category_counts"],
        "recommendation": report["recommendation"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
