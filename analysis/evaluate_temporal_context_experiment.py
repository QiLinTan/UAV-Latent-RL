"""Closed-loop and first-error analysis for B0/B1/B2 temporal ablation."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import pybullet as p
import torch
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from async_control import AsyncBuffer, AsyncScheduler, AsyncSchedulerConfig, LowLevelReferenceSelector
from envs.ForestAviary import CustomForestAviary
from envs.preprocess import preprocess_state
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from high_level_planner import (
    FUTURE_TIMES, ReferenceSequenceDecision, ReferenceSequencePacketGenerator,
    ReferenceSequencePlanner, TemporalReferenceSequencePlanner,
    append_temporal_frame, extract_environment_temporal_frame, extract_planner_input,
)
from scripts.evaluate_latent_async_pid import load_frozen_agent, make_motor_codec
from scripts.evaluate_native_latent_actor import clean_json, terminal_reason, write_json


SCENARIOS = {
    "clear": {"num_trees": 0, "route_blocking_tree": False},
    "single": {"num_trees": 0, "route_blocking_tree": True, "route_blocking_tree_count": 1, "route_tree_fraction": 0.5},
    "forest24": {"num_trees": 24, "route_blocking_tree": True, "route_blocking_tree_count": 1},
}


def load_dataset(root: Path):
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    with np.load(root / "samples.npz", allow_pickle=False) as values:
        result = {name: np.asarray(values[name]).copy() for name in values.files}
    with np.load(root / "splits.npz", allow_pickle=False) as values:
        result["splits"] = {name: np.asarray(values[name], dtype=np.int32).copy() for name in values.files}
    result["metadata"] = metadata
    return result


def load_model(variant: str, checkpoint: Path, device):
    value = torch.load(checkpoint, map_location=device, weights_only=False)
    if variant == "instant":
        model = ReferenceSequencePlanner(input_dim=27, num_points=5).to(device)
    else:
        model = TemporalReferenceSequencePlanner(variant).to(device)
    model.load_state_dict(value["model_state_dict"])
    model.eval()
    return model, value


def make_env(scenario: str, episode_len_sec: float, layout_seed: int):
    with contextlib.redirect_stdout(io.StringIO()):
        return CustomForestAviary(
            obs=ObservationType.KIN, act=ActionType.RPM, pyb_freq=240, ctrl_freq=120,
            gui=False, curriculum=False, episode_len_sec=float(episode_len_sec),
            # Tree bodies are created during environment construction, before
            # reset(seed=...).  Seed construction itself so reported episode
            # seeds actually identify fixed single/forest maps.
            seed=int(layout_seed), **SCENARIOS[scenario],
        )


def heading_angle(first, second):
    a, b = np.asarray(first, dtype=np.float64).reshape(-1), np.asarray(second, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)) / denom, -1.0, 1.0))))


def sequence_min_clearance(position, prediction, specs):
    if not specs:
        return float("inf")
    points = np.asarray(position, dtype=np.float64).reshape(1, 3) + np.asarray(prediction, dtype=np.float64)[:, :3]
    return float(min(
        np.linalg.norm(points[:, :2] - np.asarray(spec["xy"], dtype=np.float64)[None, :], axis=1).min() - float(spec["radius"]) - 0.06
        for spec in specs
    ))


class Policy:
    def __init__(self, variant, agent, planner):
        self.variant, self.agent, self.planner = variant, agent, planner
        self.history: list[np.ndarray] = []

    def decision(self, state, env):
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.agent.device).reshape(1, -1)
        with torch.no_grad():
            latent = self.agent.encoder(tensor)[0]
        latent = latent.cpu().numpy().astype(np.float32)
        current = extract_planner_input(latent, state)
        frame = extract_environment_temporal_frame(state)
        history, padding = append_temporal_frame(self.history, frame)
        t0 = time.perf_counter()
        if self.variant == "instant":
            prediction = self.planner.predict_sequence(current)
        else:
            prediction = self.planner.predict_sequence(current, history)
        inference_time = time.perf_counter() - t0
        decision = ReferenceSequenceDecision(
            position_offsets=prediction[:, :3], velocities=prediction[:, 3:],
            relative_times=FUTURE_TIMES, current_velocity=np.asarray(env.vel[0], dtype=np.float32), latent=latent,
        )
        return current, history, padding, prediction, inference_time, decision


def run_episode(scenario, seed, variant, checkpoint, latent_checkpoint, episode_len_sec, delay_max, device):
    env = make_env(scenario, episode_len_sec, seed)
    try:
        observation, _ = env.reset(seed=seed)
        state = preprocess_state(observation.reshape(-1))
        agent = load_frozen_agent(latent_checkpoint, int(state.size), device)
        planner, _ = load_model(variant, checkpoint, device)
        policy = Policy(variant, agent, planner)
        pid, codec = DSLPIDControl(drone_model=env.DRONE_MODEL), make_motor_codec(env)
        pid.reset()
        buffer, lower = AsyncBuffer(), LowLevelReferenceSelector(env.pos[0])
        generator = ReferenceSequencePacketGenerator(max_speed=0.8, max_acceleration=2.0, max_vertical_speed=0.4)
        scheduler = AsyncScheduler(AsyncSchedulerConfig(high_frequency_hz=15.0, delay_min_seconds=0.0, delay_max_seconds=delay_max, skip_probability=0.0, seed=seed), generator, buffer)
        decisions, reward_sum, path_length, tracking_sq, tracking_count = [], 0.0, 0.0, [], 0
        previous = np.asarray(env.pos[0], dtype=np.float64).copy()
        min_clearance, info, terminated, truncated, step = float("inf"), {}, False, False, 0
        while not (terminated or truncated):
            now = step * float(env.CTRL_TIMESTEP)
            snapshot, cache = state.copy(), {}
            def factory(_):
                current, history, padding, prediction, latency, decision = policy.decision(snapshot, env)
                cache.update({"current": current, "history": history, "padding": padding, "prediction": prediction, "latency": latency,
                              "position": np.asarray(env.pos[0], dtype=np.float32).copy(), "goal_distance": float(np.linalg.norm(np.asarray(env.TARGET_POS) - np.asarray(env.pos[0]))),
                              "clearance": float(env._computeNearestTreeClearance(env.pos[0]))})
                return decision
            scheduler.advance(now, factory, fallback_position=lower.last_position, fallback_velocity=np.zeros(3, dtype=np.float32))
            if cache:
                decisions.append({
                    "step": step, "time": now, "planner_input": cache["current"], "history": cache["history"],
                    "history_padding_count": cache["padding"], "prediction": cache["prediction"], "planner_inference_seconds": cache["latency"],
                    "position": cache["position"], "goal_distance": cache["goal_distance"], "clearance": cache["clearance"],
                    "predicted_sequence_min_clearance": sequence_min_clearance(cache["position"], cache["prediction"], env._tree_specs),
                })
            _, fallback = lower.select(buffer, now)
            rpm, _, _ = pid.computeControl(control_timestep=float(env.CTRL_TIMESTEP), cur_pos=np.asarray(env.pos[0], dtype=np.float64),
                cur_quat=np.asarray(p.getQuaternionFromEuler(env.rpy[0]), dtype=np.float64), cur_vel=np.asarray(env.vel[0], dtype=np.float64), cur_ang_vel=np.asarray(env.ang_v[0], dtype=np.float64),
                target_pos=np.asarray(lower.last_position, dtype=np.float64), target_vel=np.asarray(lower.last_velocity, dtype=np.float64))
            action = codec.rpm_to_normalized_action(rpm).astype(np.float32)
            observation, reward, terminated, truncated, info = env.step(action.reshape(1, -1))
            position = np.asarray(env.pos[0], dtype=np.float64)
            path_length += float(np.linalg.norm(position - previous)); previous = position.copy()
            min_clearance = min(min_clearance, float(info.get("min_tree_clearance", float("inf"))))
            tracking_sq.append(float(np.sum((position - lower.last_position) ** 2))); tracking_count += 1
            reward_sum += float(reward); state = preprocess_state(observation.reshape(-1)); step += 1
        return {"seed": seed, "success": bool(info.get("success", False)), "collision": bool(info.get("collision", False)),
            "done_reason": terminal_reason(env, info, terminated, truncated), "steps": step, "reward": reward_sum,
            "final_distance": float(np.linalg.norm(np.asarray(env.TARGET_POS) - np.asarray(env.pos[0]))), "minimum_clearance": min_clearance,
            "path_length": path_length, "tracking_rmse": float(np.sqrt(np.mean(tracking_sq))) if tracking_count else None,
            "decisions": decisions, "low_level": lower.metrics(), "scheduler": scheduler.metrics()}
    finally:
        env.close()


def feature_rows(current, history, variant):
    return np.asarray(current, dtype=np.float32) if variant == "instant" else np.concatenate((np.asarray(current, dtype=np.float32), np.asarray(history, dtype=np.float32).reshape(len(current), -1)), axis=1)


def calibrate_and_annotate(dataset, episodes, variant):
    train_rows = np.flatnonzero(np.isin(dataset["episode_id"], dataset["splits"]["train"]))
    data_features = feature_rows(dataset["planner_input"], dataset.get("environment_history"), variant)
    train_features = data_features[train_rows]
    mean, std = train_features.mean(axis=0), np.maximum(train_features.std(axis=0), 1e-3)
    scaled = (data_features - mean) / std
    train_distances = cdist(scaled[train_rows], scaled[train_rows]); np.fill_diagonal(train_distances, np.inf)
    same_episode = dataset["episode_id"][train_rows, None] == dataset["episode_id"][train_rows][None, :]
    train_distances = np.where(same_episode, np.inf, train_distances)
    train_nn = np.min(train_distances, axis=1); threshold = float(np.percentile(train_nn[np.isfinite(train_nn)], 95))
    all_student_distances, annotated = [], []
    for episode in episodes:
        rows = []
        if episode["decisions"]:
            current = np.asarray([item["planner_input"] for item in episode["decisions"]], dtype=np.float32)
            history = np.asarray([item["history"] for item in episode["decisions"]], dtype=np.float32)
            student_features = feature_rows(current, history, variant)
            distances = cdist((student_features - mean) / std, scaled[train_rows])
            nearest_order = np.argsort(distances, axis=1)[:, :8]
            nearest = train_rows[nearest_order]; nearest_distances = np.take_along_axis(distances, nearest_order, axis=1)
            for i, item in enumerate(episode["decisions"]):
                refs = dataset["reference_target"][nearest[i]]; prediction = np.asarray(item["prediction"], dtype=np.float32)
                angles = [heading_angle(refs[a, 0, :2], refs[b, 0, :2]) for a in range(8) for b in range(a + 1, 8)]
                rows.append({**{key: value for key, value in item.items() if key not in ("planner_input", "history", "prediction", "position")},
                    "nearest_train_distance": float(nearest_distances[i, 0]), "is_ood": bool(nearest_distances[i, 0] > threshold),
                    "neighbor_reference_max_heading_angle_deg": float(max(angles or [0.0])),
                    "local_observation_ambiguity": bool(nearest_distances[i, 0] <= threshold and max(angles or [0.0]) > 60.0),
                    "predicted_imminent_collision": bool(item["predicted_sequence_min_clearance"] < 0.05),
                    "pred_vs_nn_first_heading_angle_deg": heading_angle(prediction[0, :2], refs[0, 0, :2]),
                    "pred_vs_nn_first_position_delta_m": float(np.linalg.norm(prediction[0, :2] - refs[0, 0, :2]))})
                all_student_distances.append(float(nearest_distances[i, 0]))
        annotated.append({**{key: value for key, value in episode.items() if key != "decisions"}, "decisions": rows})
    return annotated, {"train_train_nearest_distance_median": float(np.median(train_nn[np.isfinite(train_nn)])), "train_train_nearest_distance_p95": threshold,
                       "student_to_train_distance_median": float(np.median(all_student_distances)), "student_to_train_distance_p95": float(np.percentile(all_student_distances, 95)),
                       "student_ood_fraction": float(np.mean(np.asarray(all_student_distances) > threshold)), "feature_space": "27D_current" if variant == "instant" else "27D_current_plus_8x16_environment_history"}


def first_failure(episode, *, require_real_history: bool = True):
    all_decisions = episode["decisions"]
    # A temporal policy receives eight repeated copies of the initial frame for
    # its first high-level decision.  Compare all variants only after the same
    # amount of observable history exists (decision eight, about 0.47 s).
    decisions = [
        item for item in all_decisions
        if not require_real_history or int(item["history_padding_count"]) == 0
    ]
    if not decisions:
        return {
            "seed": episode["seed"], "category": "insufficient_real_history",
            "first_error_step": None, "first_error_time_seconds": None,
            "history_padding_count": None, "evidence": None,
        }
    candidates = []
    for category, predicate in (("distribution_shift", lambda x: x["is_ood"]), ("observation_ambiguity", lambda x: x["local_observation_ambiguity"]), ("other", lambda x: x["predicted_imminent_collision"])):
        item = next((row for row in decisions if predicate(row)), None)
        if item is not None: candidates.append((item["step"], category, item))
    if candidates:
        _, category, item = min(candidates, key=lambda x: x[0])
    else:
        item, category = decisions[-1], "horizon_limitation"
    return {"seed": episode["seed"], "category": category, "first_error_step": item["step"], "first_error_time_seconds": item["time"],
            "history_padding_count": item["history_padding_count"], "evidence": item,
            "analysis_window": "after_full_8_frame_history" if require_real_history else "all_decisions"}


def summary(episodes):
    return {"episodes": len(episodes), "success_rate": float(np.mean([ep["success"] for ep in episodes])), "collision_rate": float(np.mean([ep["collision"] for ep in episodes])),
            "mean_final_distance": float(np.mean([ep["final_distance"] for ep in episodes])), "mean_minimum_clearance": float(np.mean([ep["minimum_clearance"] for ep in episodes])),
            "mean_path_length": float(np.mean([ep["path_length"] for ep in episodes])), "mean_tracking_rmse": float(np.mean([ep["tracking_rmse"] for ep in episodes])),
            "mean_packet_age_seconds": float(np.mean([ep["low_level"]["mean_packet_age_seconds"] for ep in episodes])),
            "mean_stale_ratio": float(np.mean([ep["low_level"]["stale_ratio"] for ep in episodes])),
            "planner_inference_ms_mean": float(np.mean([d["planner_inference_seconds"] * 1000 for ep in episodes for d in ep["decisions"]])),
            "planner_inference_ms_p95": float(np.percentile([d["planner_inference_seconds"] * 1000 for ep in episodes for d in ep["decisions"]], 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("instant", "history_stack", "gru"), required=True)
    parser.add_argument("--planner", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/latent_only_affordance_dim16/model_final")
    parser.add_argument("--episodes", type=int, default=5); parser.add_argument("--episode-len-sec", type=float, default=12.0); parser.add_argument("--delay-max-ms", type=float, default=50.0)
    args = parser.parse_args(); torch.set_num_threads(1); device = torch.device("cpu")
    dataset = load_dataset(Path(args.dataset)); model, meta = load_model(args.variant, Path(args.planner), device)
    del model
    scenarios = {}
    for index, scenario in enumerate(("clear", "single", "forest24")):
        base = 5000 + 1000 * index
        raw = [run_episode(scenario, base + offset, args.variant, Path(args.planner), args.checkpoint, args.episode_len_sec, args.delay_max_ms / 1000.0, device) for offset in range(args.episodes)]
        annotated, calibration = calibrate_and_annotate(dataset, raw, args.variant)
        failures = [first_failure(ep, require_real_history=True) for ep in annotated if not ep["success"]]
        scenarios[scenario] = {"summary": summary(annotated), "nearest_neighbor_calibration": calibration,
            "failure_category_counts": {name: int(sum(f["category"] == name for f in failures)) for name in ("distribution_shift", "observation_ambiguity", "horizon_limitation", "other", "insufficient_real_history")},
            "failure_episodes": failures, "episodes": annotated}
    result = {"experiment": "temporal_context_b0_b1_b2_v1", "variant": args.variant, "planner": str(args.planner), "planner_epoch": int(meta.get("epoch", 0)), "dataset": args.dataset,
              "logical_async_navigation_evaluation": True, "runtime_async_unchanged": True, "delay_range_seconds": [0.0, args.delay_max_ms / 1000.0], "scenarios": scenarios}
    write_json(args.output, clean_json(result))
    print(json.dumps({name: {"success": value["summary"]["success_rate"], "collision": value["summary"]["collision_rate"], "ood": value["nearest_neighbor_calibration"]["student_ood_fraction"], "failures": value["failure_category_counts"]} for name, value in scenarios.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
