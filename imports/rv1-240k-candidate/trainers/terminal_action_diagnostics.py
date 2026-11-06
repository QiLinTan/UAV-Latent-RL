"""Evaluation-only diagnostics for terminal action scaling and critic values."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


XY_SCAN_SPEEDS = np.asarray((0.0, 0.1, 0.2, 0.4, 0.6, 0.8), dtype=np.float32)


def xy_action_candidates(
    policy_action,
    *,
    position,
    target_position,
    max_planar_speed: float,
    scan_speeds=XY_SCAN_SPEEDS,
) -> tuple[np.ndarray, list[dict]]:
    """Build normalized XY actions while preserving the actor's z action."""

    policy = np.asarray(policy_action, dtype=np.float32).reshape(-1)
    if policy.size != 3:
        raise ValueError("XY action Q scan requires a 3-D policy action.")
    if max_planar_speed <= 0.0:
        raise ValueError("max_planar_speed must be positive.")
    speeds = np.asarray(scan_speeds, dtype=np.float32).reshape(-1)
    if speeds.size == 0 or np.any(speeds < 0.0) or np.any(speeds > max_planar_speed + 1e-6):
        raise ValueError("scan_speeds must lie in [0, max_planar_speed].")

    position = np.asarray(position, dtype=np.float32).reshape(3)
    target = np.asarray(target_position, dtype=np.float32).reshape(3)
    goal_xy = target[:2] - position[:2]
    goal_norm = float(np.linalg.norm(goal_xy))
    goal_direction = (
        goal_xy / goal_norm if goal_norm > 1e-8 else np.array([1.0, 0.0], dtype=np.float32)
    )
    actor_xy = policy[:2]
    actor_norm = float(np.linalg.norm(actor_xy))
    actor_direction = actor_xy / actor_norm if actor_norm > 1e-8 else goal_direction.copy()

    actions = []
    metadata = []
    for direction_name, direction in (
        ("goal", goal_direction),
        ("actor", actor_direction),
    ):
        for speed in speeds:
            if speed == 0.0 and direction_name == "actor":
                continue
            action = policy.copy()
            action[:2] = direction * (float(speed) / float(max_planar_speed))
            actions.append(np.clip(action, -1.0, 1.0))
            metadata.append(
                {
                    "kind": "speed_scan",
                    "direction": "stop" if speed == 0.0 else direction_name,
                    "speed_mps": float(speed),
                }
            )

    actions.append(policy.copy())
    metadata.append(
        {
            "kind": "actor_raw",
            "direction": "actor_raw",
            "speed_mps": float(min(actor_norm, 1.0) * max_planar_speed),
        }
    )
    return np.asarray(actions, dtype=np.float32), metadata


def summarize_action_interface(records: list[dict], max_planar_speed: float) -> dict:
    if not records:
        return {"record_count": 0}
    raw_norm = np.asarray([item["actor_xy_norm"] for item in records], dtype=np.float64)
    command_speed = np.asarray([item["cmd_xy_speed"] for item in records], dtype=np.float64)
    actual_speed = np.asarray([item["actual_xy_speed"] for item in records], dtype=np.float64)
    expected = np.minimum(raw_norm, 1.0) * float(max_planar_speed)
    error = command_speed - expected
    return {
        "record_count": int(len(records)),
        "episode_count": int(len({int(item["episode"]) for item in records})),
        "max_planar_speed": float(max_planar_speed),
        "actor_xy_norm_mean": float(np.mean(raw_norm)),
        "actor_xy_norm_median": float(np.median(raw_norm)),
        "actor_xy_norm_p95": float(np.percentile(raw_norm, 95.0)),
        "actor_xy_norm_at_least_0_95_fraction": float(np.mean(raw_norm >= 0.95)),
        "cmd_xy_speed_mean": float(np.mean(command_speed)),
        "actual_xy_speed_mean": float(np.mean(actual_speed)),
        "mapping_mae": float(np.mean(np.abs(error))),
        "mapping_max_abs_error": float(np.max(np.abs(error))),
        "mapping_is_linear_below_unit_norm": bool(np.max(np.abs(error)) <= 1e-5),
    }


def summarize_xy_q_scan(records: list[dict]) -> dict:
    grouped = defaultdict(list)
    best_counts = Counter()
    spreads = []
    actor_gaps = []
    for record in records:
        candidates = record["candidates"]
        q_values = np.asarray([item["q_min"] for item in candidates], dtype=np.float64)
        if q_values.size:
            spreads.append(float(np.max(q_values) - np.min(q_values)))
            best = candidates[int(np.argmax(q_values))]
            best_counts[f"{best['direction']}:{best['speed_mps']:.1f}"] += 1
        actor_q = next(
            (float(item["q_min"]) for item in candidates if item["kind"] == "actor_raw"),
            np.nan,
        )
        if q_values.size and np.isfinite(actor_q):
            actor_gaps.append(float(np.max(q_values) - actor_q))
        for item in candidates:
            if item["kind"] != "speed_scan":
                continue
            grouped[(item["direction"], float(item["speed_mps"]))].append(
                float(item["q_min"])
            )

    aggregate = []
    for (direction, speed), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        aggregate.append(
            {
                "direction": direction,
                "speed_mps": speed,
                "count": int(len(values)),
                "q_min_mean": float(np.mean(values)),
                "q_min_std": float(np.std(values)),
                "q_min_median": float(np.median(values)),
            }
        )
    return {
        "state_count": int(len(records)),
        "episode_count": int(len({int(item["episode"]) for item in records})),
        "mean_within_state_q_range": float(np.mean(spreads)) if spreads else None,
        "median_within_state_q_range": float(np.median(spreads)) if spreads else None,
        "mean_best_minus_actor_q": float(np.mean(actor_gaps)) if actor_gaps else None,
        "best_candidate_counts": dict(best_counts),
        "aggregate": aggregate,
    }


def export_action_interface_audit(
    output_dir,
    *,
    records: list[dict],
    synthetic_mapping: list[dict],
    max_planar_speed: float,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    summary = summarize_action_interface(records, max_planar_speed)
    payload = {
        "schema_version": "upper_action_interface_audit_v1",
        "summary": summary,
        "synthetic_mapping": synthetic_mapping,
        "records": records,
    }
    json_path = os.path.join(output_dir, "action_interface_audit.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    if records:
        raw_norm = np.asarray([item["actor_xy_norm"] for item in records])
        command_speed = np.asarray([item["cmd_xy_speed"] for item in records])
        actual_speed = np.asarray([item["actual_xy_speed"] for item in records])
        goal_distance = np.asarray([item["goal_distance_before"] for item in records])
        x_line = np.linspace(0.0, max(1.05, float(np.max(raw_norm))), 200)
        y_line = np.minimum(x_line, 1.0) * float(max_planar_speed)

        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        ax.scatter(raw_norm, command_speed, s=8, alpha=0.28, label="mapped command")
        ax.plot(x_line, y_line, "k--", linewidth=1.5, label="expected mapping")
        ax.set_xlabel("raw actor planar norm ||a_xy||")
        ax.set_ylabel("command planar speed ||v_xy_cmd|| [m/s]")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "action_norm_to_cmd_speed.png"), dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        ax.scatter(goal_distance, command_speed, s=10, alpha=0.35, label="command speed")
        ax.scatter(goal_distance, actual_speed, s=10, alpha=0.35, label="K1 actual speed")
        ax.set_xlabel("goal distance before decision [m]")
        ax.set_ylabel("planar speed [m/s]")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "t1_speed_vs_goal_distance.png"), dpi=180)
        plt.close(fig)
    return {"json": json_path, "summary": summary}


def export_xy_q_scan(output_dir, *, records: list[dict]) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    summary = summarize_xy_q_scan(records)
    payload = {
        "schema_version": "upper_terminal_xy_critic_q_scan_v1",
        "filters": {
            "timeout_class": "T1",
            "goal_distance_open_interval_m": [0.3, 1.2],
            "minimum_tree_clearance_m": 0.35,
        },
        "summary": summary,
        "records": records,
    }
    json_path = os.path.join(output_dir, "terminal_xy_critic_q_scan.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    aggregate = summary.get("aggregate", [])
    if aggregate:
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        for direction, marker in (("goal", "o"), ("actor", "s"), ("stop", "x")):
            rows = [item for item in aggregate if item["direction"] == direction]
            if not rows:
                continue
            ax.errorbar(
                [item["speed_mps"] for item in rows],
                [item["q_min_mean"] for item in rows],
                yerr=[item["q_min_std"] for item in rows],
                marker=marker,
                capsize=3,
                label=direction,
            )
        ax.set_xlabel("candidate planar speed [m/s]")
        ax.set_ylabel("min(Q1, Q2)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "terminal_xy_q_vs_speed.png"), dpi=180)
        plt.close(fig)
    return {"json": json_path, "summary": summary}
