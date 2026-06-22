from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ACTION_NAMES = ("vx", "vy", "vz", "yaw_rate")
OBS_GOAL_DELTA_SLICE = slice(13, 16)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite_array(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def stats(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    array = finite_array(list(values))
    if array.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "median": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def bool_from_csv(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def vector_rows(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return np.empty((0, 4), dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def per_dim_stats(array: np.ndarray, *, saturation_threshold: float = 0.95) -> dict[str, Any]:
    if array.size == 0:
        return {
            name: {"count": 0, "mean": None, "std": None, "min": None, "max": None, "saturation_percentage": None}
            for name in ACTION_NAMES
        }
    result = {}
    for index, name in enumerate(ACTION_NAMES):
        column = array[:, index]
        result[name] = {
            **stats(column),
            "saturation_percentage": float(np.mean(np.abs(column) >= saturation_threshold) * 100.0),
        }
    return result


def corrcoef_or_none(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return None
    aa = a[mask]
    bb = b[mask]
    if float(np.std(aa)) == 0.0 or float(np.std(bb)) == 0.0:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def episode_groups(steps: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in steps:
        groups[int(row["episode_index"])].append(row)
    return dict(sorted(groups.items()))


def reward_breakdown(steps: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    reward_cfg = config["reward_done_config"]
    progress_reward = np.asarray(
        [float(row.get("progress", 0.0)) * float(reward_cfg["progress_scale"]) for row in steps],
        dtype=np.float64,
    )
    height_penalty = np.asarray([float(row.get("height_penalty", 0.0)) for row in steps], dtype=np.float64)
    vertical_penalty = np.asarray(
        [float(row.get("vertical_velocity_penalty", 0.0)) for row in steps],
        dtype=np.float64,
    )
    z_penalty = np.asarray([float(row.get("z_action_penalty", 0.0)) for row in steps], dtype=np.float64)
    normalized_actions = vector_rows(steps, "normalized_action")
    action_penalty = (
        float(reward_cfg["action_penalty_scale"]) * np.sum(normalized_actions**2, axis=1)
        if normalized_actions.size
        else np.asarray([], dtype=np.float64)
    )
    reward = np.asarray([float(row.get("reward", 0.0)) for row in steps], dtype=np.float64)
    nonterminal = np.asarray([str(row.get("done_reason", "running")) == "running" for row in steps])
    total_abs_regularization = height_penalty + vertical_penalty + z_penalty
    if action_penalty.size == total_abs_regularization.size:
        total_abs_regularization = total_abs_regularization + action_penalty
    denominator = float(np.sum(np.abs(progress_reward)) + np.sum(total_abs_regularization))
    return {
        "progress_reward": stats(progress_reward),
        "height_penalty": stats(height_penalty),
        "vertical_velocity_penalty": stats(vertical_penalty),
        "z_action_penalty": stats(z_penalty),
        "action_penalty": stats(action_penalty),
        "reward": stats(reward),
        "sum_progress_reward": float(progress_reward.sum()),
        "sum_abs_progress_reward": float(np.abs(progress_reward).sum()),
        "sum_regularization_penalties": float(total_abs_regularization.sum()),
        "progress_abs_share_of_progress_plus_regularization": (
            float(np.abs(progress_reward).sum() / denominator) if denominator > 0.0 else None
        ),
        "mean_running_reward": float(reward[nonterminal].mean()) if nonterminal.any() else None,
        "timeout_penalty_count": int(sum(str(row.get("done_reason")) == "timeout" for row in steps)),
    }


def first_last_window(groups: dict[int, list[dict[str, Any]]], window: int = 20) -> dict[str, Any]:
    first_rows: list[dict[str, Any]] = []
    last_rows: list[dict[str, Any]] = []
    for rows in groups.values():
        first_rows.extend(rows[:window])
        last_rows.extend(rows[-window:])

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "row_count": len(rows),
            "progress": stats([float(row["progress"]) for row in rows]),
            "distance_to_goal": stats([float(row["distance_to_goal"]) for row in rows]),
            "height": stats([float(row["height"]) for row in rows]),
            "vertical_velocity": stats([float(row["vertical_velocity"]) for row in rows]),
            "scaled_action": per_dim_stats(vector_rows(rows, "scaled_velocity_command"), saturation_threshold=0.95),
            "normalized_action": per_dim_stats(vector_rows(rows, "normalized_action"), saturation_threshold=0.95),
            "raw_actor_action": per_dim_stats(vector_rows(rows, "raw_actor_action"), saturation_threshold=0.95),
        }

    return {
        "first_20_steps_per_episode": summarize(first_rows),
        "last_20_steps_per_episode": summarize(last_rows),
    }


def analyze(run_dir: Path) -> dict[str, Any]:
    config = read_json(run_dir / "config.json")
    summary = read_json(run_dir / "summary.json")
    episodes = read_csv(run_dir / "episodes.csv")
    steps = read_jsonl(run_dir / "steps.jsonl")
    groups = episode_groups(steps)

    initial_distances = np.asarray([float(row["initial_distance"]) for row in episodes], dtype=np.float64)
    final_distances = np.asarray([float(row["final_distance"]) for row in episodes], dtype=np.float64)
    distance_delta = np.asarray([float(row["distance_improvement"]) for row in episodes], dtype=np.float64)
    done_reasons = Counter(row["done_reason"] for row in episodes)
    collisions = sum(bool_from_csv(row["collision"]) for row in episodes)

    positions = np.asarray([row["position"] for row in steps], dtype=np.float64)
    velocities = np.asarray([row["velocity"] for row in steps], dtype=np.float64)
    scaled_actions = vector_rows(steps, "scaled_velocity_command")
    normalized_actions = vector_rows(steps, "normalized_action")
    actor_steps = [row for row in steps if row.get("raw_actor_action") is not None]
    raw_actor_actions = vector_rows(actor_steps, "raw_actor_action")
    actor_progress = np.asarray([float(row["progress"]) for row in actor_steps], dtype=np.float64)
    progress = np.asarray([float(row["progress"]) for row in steps], dtype=np.float64)
    distances = np.asarray([float(row["distance_to_goal"]) for row in steps], dtype=np.float64)
    heights = np.asarray([float(row["height"]) for row in steps], dtype=np.float64)
    vertical_velocity = np.asarray([float(row["vertical_velocity"]) for row in steps], dtype=np.float64)
    states = np.asarray([row["state"] for row in steps], dtype=np.float64)
    next_states = np.asarray([row["next_state"] for row in steps], dtype=np.float64)
    goal_delta_state = states[:, OBS_GOAL_DELTA_SLICE]
    goal_delta_next = next_states[:, OBS_GOAL_DELTA_SLICE]
    goal_distance_from_delta = np.linalg.norm(goal_delta_next, axis=1)
    distance_consistency = np.abs(goal_distance_from_delta - distances)
    xy_motion = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1) if len(positions) > 1 else np.asarray([])

    action_progress_correlation = {}
    for index, name in enumerate(ACTION_NAMES):
        action_progress_correlation[name] = {
            "scaled_action_vs_progress": corrcoef_or_none(scaled_actions[:, index], progress)
            if scaled_actions.size
            else None,
            "normalized_action_vs_progress": corrcoef_or_none(normalized_actions[:, index], progress)
            if normalized_actions.size
            else None,
            "raw_actor_action_vs_progress": None,
        }
        if raw_actor_actions.size:
            action_progress_correlation[name]["raw_actor_action_vs_progress"] = corrcoef_or_none(
                raw_actor_actions[:, index],
                actor_progress,
            )
    velocity_progress_correlation = {
        "vx_vs_progress": corrcoef_or_none(velocities[:, 0], progress),
        "vy_vs_progress": corrcoef_or_none(velocities[:, 1], progress),
        "vz_vs_progress": corrcoef_or_none(velocities[:, 2], progress),
    }

    active_scaled = {
        name: float(np.nanstd(scaled_actions[:, index])) if scaled_actions.size else None
        for index, name in enumerate(ACTION_NAMES)
    }
    most_active_scaled = max(active_scaled, key=lambda key: active_scaled[key] or -1.0)
    hover_like_episodes = [
        int(row["episode_index"])
        for row in episodes
        if abs(float(row["distance_improvement"])) < 0.05
        and float(row["max_abs_vertical_velocity"]) < 0.02
        and not bool_from_csv(row["collision"])
    ]

    distance_curves = {}
    for episode, rows in groups.items():
        sample_indices = sorted(
            set(
                [0, len(rows) - 1]
                + [min(len(rows) - 1, idx) for idx in range(19, len(rows), 20)]
            )
        )
        distance_curves[str(episode)] = [
            {
                "episode_step": int(rows[index]["episode_step"]),
                "distance_to_goal": float(rows[index]["distance_to_goal"]),
                "progress": float(rows[index]["progress"]),
                "position": rows[index]["position"],
            }
            for index in sample_indices
        ]

    return {
        "run_dir": str(run_dir),
        "config": config,
        "summary": summary,
        "episode_count": len(episodes),
        "episode_length": stats([float(row["episode_length"]) for row in episodes]),
        "return": stats([float(row["episode_return"]) for row in episodes]),
        "done_reason_counts": dict(sorted(done_reasons.items())),
        "collision_count": int(collisions),
        "initial_distance": stats(initial_distances),
        "final_distance": stats(final_distances),
        "distance_delta": stats(distance_delta),
        "positive_progress_episode_count": int(np.sum(distance_delta > 0.0)),
        "positive_progress_episode_ratio": float(np.mean(distance_delta > 0.0)) if len(distance_delta) else None,
        "mean_progress_per_step": stats(progress),
        "distance_curve_samples": distance_curves,
        "height": stats(heights),
        "vertical_velocity": stats(vertical_velocity),
        "raw_action": per_dim_stats(raw_actor_actions, saturation_threshold=float(config.get("saturation_threshold", 0.95))),
        "normalized_action": per_dim_stats(normalized_actions, saturation_threshold=float(config.get("saturation_threshold", 0.95))),
        "scaled_action": per_dim_stats(scaled_actions, saturation_threshold=0.95),
        "most_active_scaled_action_dimension": most_active_scaled,
        "reward_breakdown": reward_breakdown(steps, config),
        "first_last_20_step_comparison": first_last_window(groups),
        "observation_goal_delta": {
            "slice": [OBS_GOAL_DELTA_SLICE.start, OBS_GOAL_DELTA_SLICE.stop],
            "initial_samples": goal_delta_state[:5].tolist(),
            "next_samples": goal_delta_next[:5].tolist(),
            "mean_goal_delta": goal_delta_next.mean(axis=0).tolist(),
            "distance_consistency_abs_error": stats(distance_consistency),
            "interpretation": (
                "Observation indices 13:16 contain goal_position - position. "
                "For this run the target is mostly +x from reset, so positive vx is the direct "
                "low-level command to test before more training."
            ),
        },
        "motion_and_frame_diagnostics": {
            "mean_position": positions.mean(axis=0).tolist(),
            "final_position_mean_by_episode": [
                groups[idx][-1]["position"] for idx in sorted(groups)
            ],
            "xy_motion_per_step": stats(xy_motion),
            "action_progress_correlation": action_progress_correlation,
            "velocity_progress_correlation": velocity_progress_correlation,
            "frame_inference": (
                "The ROS launch sets velocity_estimate_in_world_frame=false in autopilot, "
                "so command frame needs live probing. Because reset yaw is near zero and goal_delta "
                "is +x, constant +vx and goal-direction probes are the next required test."
            ),
        },
        "hover_without_progress": {
            "episode_count": len(hover_like_episodes),
            "episode_indices": hover_like_episodes,
            "criterion": "abs(distance_delta) < 0.05 and max_abs_vertical_velocity < 0.02 without collision",
        },
        "diagnosis": {
            "most_likely_cause": (
                "The policy learned stable low-action flight but did not learn to command +vx toward "
                "the +x goal. The strongest actor bias is +vy, while mean vx is slightly negative."
            ),
            "coordinate_or_action_mapping_suspicion": (
                "Still open. The observation goal delta is present and numerically consistent, but "
                "the live action frame must be verified with a hand-coded goal-direction policy."
            ),
            "reward_weight_suspicion": (
                "High. The absolute progress signal is tiny compared with z-action/action regularization, "
                "so hovering or sideways low-risk commands can dominate early learning."
            ),
            "task_difficulty_suspicion": (
                "Moderate. A 5 m goal over 200 low-speed steps is not extreme, but it is too complex "
                "to debug before proving hand-coded goal-direction progress."
            ),
            "stage1_needed": True,
            "do_not_start_latent": True,
        },
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    rb = result["reward_breakdown"]
    obs = result["observation_goal_delta"]
    diag = result["diagnosis"]
    lines = [
        "# AvoidBench Navigation Smoke Analysis",
        "",
        f"Run: `{result['run_dir']}`",
        "",
        "## Executive conclusion",
        "",
        f"- Episodes: {result['episode_count']}",
        f"- Done reasons: `{json.dumps(result['done_reason_counts'], sort_keys=True)}`",
        f"- Mean distance delta: `{result['distance_delta']['mean']:.6f} m`",
        f"- Positive-progress episodes: `{result['positive_progress_episode_count']}/{result['episode_count']}`",
        f"- Most active scaled action dimension: `{result['most_active_scaled_action_dimension']}`",
        f"- Mean raw actor action vx/vy: `{result['raw_action']['vx']['mean']:.6f}` / `{result['raw_action']['vy']['mean']:.6f}`",
        "",
        "The run is stable but not a navigation baseline. It mostly hovers near",
        "the reset height while commanding sideways velocity more strongly than",
        "goal-directed `+vx`.",
        "",
        "## Episode and distance statistics",
        "",
        f"- episode length mean/min/max: `{result['episode_length']['mean']}` / `{result['episode_length']['min']}` / `{result['episode_length']['max']}`",
        f"- return mean/min/max: `{result['return']['mean']}` / `{result['return']['min']}` / `{result['return']['max']}`",
        f"- initial distance mean: `{result['initial_distance']['mean']}`",
        f"- final distance mean: `{result['final_distance']['mean']}`",
        f"- distance delta mean/min/max: `{result['distance_delta']['mean']}` / `{result['distance_delta']['min']}` / `{result['distance_delta']['max']}`",
        f"- per-step progress mean/std: `{result['mean_progress_per_step']['mean']}` / `{result['mean_progress_per_step']['std']}`",
        "",
        "## Height and safety",
        "",
        f"- collision count: `{result['collision_count']}`",
        f"- height mean/min/max: `{result['height']['mean']}` / `{result['height']['min']}` / `{result['height']['max']}`",
        f"- vertical velocity mean/std/min/max: `{result['vertical_velocity']['mean']}` / `{result['vertical_velocity']['std']}` / `{result['vertical_velocity']['min']}` / `{result['vertical_velocity']['max']}`",
        f"- stable-hover-without-progress episodes: `{result['hover_without_progress']['episode_count']}`",
        "",
        "## Action diagnostics",
        "",
        "| dimension | raw mean | raw std | scaled mean | scaled std | normalized saturation |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ACTION_NAMES:
        raw = result["raw_action"][name]
        scaled = result["scaled_action"][name]
        norm = result["normalized_action"][name]
        lines.append(
            f"| {name} | {raw['mean']:.6f} | {raw['std']:.6f} | "
            f"{scaled['mean']:.6f} | {scaled['std']:.6f} | {norm['saturation_percentage']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "The actor does not saturate. The main issue is direction: the learned",
            "mean actor output is biased toward `+vy`, while the goal delta indicates",
            "the target is primarily in `+x`.",
            "",
            "## Reward balance",
            "",
            f"- sum progress reward: `{rb['sum_progress_reward']}`",
            f"- sum abs progress reward: `{rb['sum_abs_progress_reward']}`",
            f"- sum regularization penalties: `{rb['sum_regularization_penalties']}`",
            f"- abs progress share of progress plus regularization: `{rb['progress_abs_share_of_progress_plus_regularization']}`",
            f"- z-action penalty mean: `{rb['z_action_penalty']['mean']}`",
            f"- action penalty mean: `{rb['action_penalty']['mean']}`",
            "",
            "The progress signal is very small. Regularization, especially z-action",
            "and action penalties, is large enough to make low-motion behavior an",
            "attractive early solution.",
            "",
            "## Observation and frame checks",
            "",
            f"- goal delta observation slice: `{obs['slice']}`",
            f"- mean goal delta: `{obs['mean_goal_delta']}`",
            f"- distance consistency abs error max: `{obs['distance_consistency_abs_error']['max']}`",
            "",
            obs["interpretation"],
            "",
            "The goal delta is present and numerically consistent with",
            "`distance_to_goal`. The unresolved question is command frame: the launch",
            "sets `velocity_estimate_in_world_frame=false`, so a live hand-coded",
            "policy must verify whether world-frame or body-frame goal direction",
            "actually reduces distance.",
            "",
            "## Front/back 20-step comparison",
            "",
            f"- first-window progress mean: `{result['first_last_20_step_comparison']['first_20_steps_per_episode']['progress']['mean']}`",
            f"- last-window progress mean: `{result['first_last_20_step_comparison']['last_20_steps_per_episode']['progress']['mean']}`",
            "",
            "There is no clear late-episode improvement. The policy remains stable",
            "but does not increasingly point toward the goal.",
            "",
            "## Diagnosis",
            "",
            f"- most likely cause: {diag['most_likely_cause']}",
            f"- coordinate/action mapping suspicion: {diag['coordinate_or_action_mapping_suspicion']}",
            f"- reward weight suspicion: {diag['reward_weight_suspicion']}",
            f"- task difficulty suspicion: {diag['task_difficulty_suspicion']}",
            "",
            "## Decision",
            "",
            "Do not continue longer navigation training yet. Run the",
            "`constant-goal-direction` hand-coded policy probe first. If that probe",
            "cannot reliably lower distance, fix action frame, goal delta, or distance",
            "calculation before Stage 1 training. Do not start latent work.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze AvoidBench navigation smoke diagnostics.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/avoidbench_plain_td3_smoke/20260609-141223-navigation_smoke"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("docs/avoidbench_navigation_smoke_analysis.md"),
    )
    args = parser.parse_args()
    summary_output = args.summary_output or args.run_dir / "navigation_analysis_summary.json"
    result = analyze(args.run_dir)
    summary_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_report(args.report_output, result)
    print(json.dumps({
        "summary_output": str(summary_output),
        "report_output": str(args.report_output),
        "mean_distance_delta": result["distance_delta"]["mean"],
        "positive_progress": [
            result["positive_progress_episode_count"],
            result["episode_count"],
        ],
        "stage1_needed": result["diagnosis"]["stage1_needed"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
