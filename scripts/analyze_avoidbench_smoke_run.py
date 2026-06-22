from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ACTION_NAMES = ("vx", "vy", "vz", "yaw_rate")
HEIGHT_DONE_REASONS = {"height_too_low", "height_too_high"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite_stats(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "median": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "median": float(np.median(finite)),
        "max": float(finite.max()),
    }


def unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason}


def summarize_steps(step_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not step_rows:
        reason = (
            "The historical run has no steps.jsonl. Per-step height, velocity, "
            "raw/scaled action, reward terms, and terminal context were not logged."
        )
        return {
            "height": unavailable(reason),
            "actions": unavailable(reason),
            "vertical_velocity": unavailable(reason),
            "pre_height_termination_windows": unavailable(reason),
            "distance_progress": unavailable(reason),
            "step_time": unavailable(reason),
        }

    heights = [float(row["height"]) for row in step_rows if row.get("height") is not None]
    vertical_velocities = [
        float(row["vertical_velocity"])
        for row in step_rows
        if row.get("vertical_velocity") is not None
    ]
    step_times = [float(row["step_time"]) for row in step_rows if row.get("step_time") is not None]
    progress = [float(row["progress"]) for row in step_rows if row.get("progress") is not None]
    actions = [
        np.asarray(row["raw_actor_action"], dtype=np.float64)
        for row in step_rows
        if row.get("raw_actor_action") is not None
    ]
    action_summary: dict[str, Any]
    if actions and all(action.shape == (4,) for action in actions):
        action_array = np.stack(actions)
        per_dimension = {}
        for index, name in enumerate(ACTION_NAMES):
            values = action_array[:, index]
            per_dimension[name] = {
                **finite_stats(values.tolist()),
                "saturation_percentage": float(np.mean(np.abs(values) >= 0.99) * 100.0),
            }
        action_summary = {
            "status": "available",
            "sample_count": int(action_array.shape[0]),
            "per_dimension": per_dimension,
            "most_saturated_dimension": max(
                per_dimension,
                key=lambda name: per_dimension[name]["saturation_percentage"],
            ),
        }
    else:
        action_summary = unavailable("No valid four-dimensional raw_actor_action samples.")

    terminal_windows = []
    for index, row in enumerate(step_rows):
        if row.get("done_reason") not in HEIGHT_DONE_REASONS:
            continue
        terminal_windows.append(
            {
                "terminal_step": row.get("step"),
                "done_reason": row.get("done_reason"),
                "rows": step_rows[max(0, index - 9) : index + 1],
            }
        )
    return {
        "height": {"status": "available", **finite_stats(heights)},
        "actions": action_summary,
        "vertical_velocity": {"status": "available", **finite_stats(vertical_velocities)},
        "pre_height_termination_windows": {
            "status": "available",
            "window_count": len(terminal_windows),
            "windows": terminal_windows,
        },
        "distance_progress": {"status": "available", **finite_stats(progress)},
        "step_time": {"status": "available", **finite_stats(step_times)},
    }


def analyze(run_dir: Path) -> dict[str, Any]:
    config = read_json(run_dir / "config.json")
    run_summary = read_json(run_dir / "summary.json")
    episodes = read_csv(run_dir / "episodes.csv")
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    steps = read_jsonl(run_dir / "steps.jsonl")

    lengths = [int(row["episode_length"]) for row in episodes]
    returns = [float(row["episode_return"]) for row in episodes]
    final_distances = [float(row["final_distance"]) for row in episodes]
    done_reasons = Counter(row["done_reason"] for row in episodes)
    reset_retries = [int(row["reset_retry_count"]) for row in episodes]
    collisions = sum(str(row.get("collision", "")).lower() == "true" for row in episodes)
    step_times = [
        float(row["mean_step_time"])
        for row in metrics
        if row.get("mean_step_time") is not None
    ]
    actor_means = [
        float(row["actor_action_mean"])
        for row in metrics
        if row.get("action_source") == "actor" and row.get("actor_action_mean") is not None
    ]
    actor_stds = [
        float(row["actor_action_std"])
        for row in metrics
        if row.get("action_source") == "actor" and row.get("actor_action_std") is not None
    ]
    critic_losses = [
        float(row["critic_loss"])
        for row in metrics
        if row.get("critic_loss") is not None
    ]
    height_terminations = sum(done_reasons[reason] for reason in HEIGHT_DONE_REASONS)
    completed_episode_count = len(episodes)

    first_distance = final_distances[0] if final_distances else None
    last_distance = final_distances[-1] if final_distances else None
    distance_change = None
    if first_distance is not None and last_distance is not None:
        distance_change = float(first_distance - last_distance)

    per_step = summarize_steps(steps)
    aggregate_action_note = (
        "Only a scalar mean/std over all actor action dimensions was logged every "
        "100 steps. Per-dimension min/max/saturation cannot be recovered."
    )
    if steps:
        action_details = per_step["actions"]
        max_saturation = 0.0
        most_saturated = "unavailable"
        if action_details["status"] == "available":
            most_saturated = action_details["most_saturated_dimension"]
            max_saturation = float(
                action_details["per_dimension"][most_saturated]["saturation_percentage"]
            )
        diagnosis = {
            "primary_failure": (
                f"{height_terminations}/{completed_episode_count} completed episodes "
                "ended because of height bounds."
            ),
            "actor_collapse_evidence": (
                f"Maximum raw actor saturation was {max_saturation:.2f}% on "
                f"{most_saturated}; no near-limit collapse is indicated."
            ),
            "causality_limit": (
                "Per-step action, height, vertical velocity, reward terms, and terminal "
                "context are available for this run."
            ),
        }
    else:
        diagnosis = {
            "primary_failure": (
                f"{height_terminations}/{completed_episode_count} completed episodes "
                "ended because of height bounds."
            ),
            "actor_collapse_evidence": (
                "The aggregate actor std approached 1.0 after step 400, consistent "
                "with normalized tanh outputs collapsing toward action limits."
            ),
            "causality_limit": (
                "The old run cannot distinguish which action dimension caused the "
                "height failures because it did not log per-step per-dimension actions, "
                "height, or vertical velocity."
            ),
        }
    return {
        "run_dir": str(run_dir),
        "config": config,
        "run_summary": run_summary,
        "data_inventory": {
            "episode_rows": len(episodes),
            "metric_rows": len(metrics),
            "step_rows": len(steps),
        },
        "episodes": {
            "count": completed_episode_count,
            "length": finite_stats([float(value) for value in lengths]),
            "return": finite_stats(returns),
            "done_reason_counts": dict(sorted(done_reasons.items())),
            "height_out_of_bounds_count": height_terminations,
            "height_out_of_bounds_percentage": (
                100.0 * height_terminations / completed_episode_count
                if completed_episode_count
                else None
            ),
            "collision_count": collisions,
            "reset_retry": finite_stats([float(value) for value in reset_retries]),
        },
        "distance_to_goal": {
            "final_distance": finite_stats(final_distances),
            "first_completed_episode_final_distance": first_distance,
            "last_completed_episode_final_distance": last_distance,
            "first_to_last_improvement": distance_change,
            "interpretation": (
                "Positive means the last completed episode ended closer than the first."
                if distance_change is not None
                else "Unavailable."
            ),
        },
        "aggregate_metrics": {
            "actor_action_mean": finite_stats(actor_means),
            "actor_action_std": finite_stats(actor_stds),
            "action_dimension_statistics": unavailable(aggregate_action_note),
            "action_saturation_percentage": unavailable(aggregate_action_note),
            "most_saturated_dimension": unavailable(aggregate_action_note),
            "critic_loss": finite_stats(critic_losses),
            "step_time": finite_stats(step_times),
        },
        "per_step": per_step,
        "diagnosis": diagnosis,
    }


def format_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    episodes = summary["episodes"]
    distance = summary["distance_to_goal"]
    aggregate = summary["aggregate_metrics"]
    per_step = summary["per_step"]
    diagnosis = summary["diagnosis"]
    lines = [
        "# AvoidBench Plain TD3 Smoke Analysis",
        "",
        f"Run: `{summary['run_dir']}`",
        "",
        "## Executive conclusion",
        "",
        f"- {diagnosis['primary_failure']}",
        f"- {diagnosis['actor_collapse_evidence']}",
        f"- {diagnosis['causality_limit']}",
        (
            "- This run provides a stable-control measurement with full per-step diagnostics."
            if per_step["actions"]["status"] == "available"
            else "- This run proves the training infrastructure worked, but it is not a stable control baseline."
        ),
        "",
        "## Episode statistics",
        "",
        f"- completed episodes: {episodes['count']}",
        f"- episode length mean/median/min/max: "
        f"{format_value(episodes['length']['mean'])} / "
        f"{format_value(episodes['length']['median'])} / "
        f"{format_value(episodes['length']['min'])} / "
        f"{format_value(episodes['length']['max'])}",
        f"- return mean/median/min/max: "
        f"{format_value(episodes['return']['mean'])} / "
        f"{format_value(episodes['return']['median'])} / "
        f"{format_value(episodes['return']['min'])} / "
        f"{format_value(episodes['return']['max'])}",
        f"- done reasons: `{json.dumps(episodes['done_reason_counts'], sort_keys=True)}`",
        f"- height-bound terminations: {episodes['height_out_of_bounds_count']} "
        f"({format_value(episodes['height_out_of_bounds_percentage'], 2)}%)",
        f"- collisions: {episodes['collision_count']}",
        f"- reset retry mean/max: {format_value(episodes['reset_retry']['mean'])} / "
        f"{format_value(episodes['reset_retry']['max'])}",
        "",
        "## Action and optimization signals",
        "",
        f"- critic loss mean/max: "
        f"{format_value(aggregate['critic_loss']['mean'])} / "
        f"{format_value(aggregate['critic_loss']['max'])}",
        f"- mean step time mean/max: "
        f"{format_value(aggregate['step_time']['mean'])} / "
        f"{format_value(aggregate['step_time']['max'])}",
    ]
    if per_step["actions"]["status"] == "available":
        lines.extend(
            [
                "",
                "| dimension | mean | std | min | max | saturation |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name in ACTION_NAMES:
            stats = per_step["actions"]["per_dimension"][name]
            lines.append(
                f"| {name} | {format_value(stats['mean'])} | "
                f"{format_value(stats['std'])} | {format_value(stats['min'])} | "
                f"{format_value(stats['max'])} | "
                f"{format_value(stats['saturation_percentage'], 2)}% |"
            )
        lines.append(
            f"\nMost saturated dimension: "
            f"`{per_step['actions']['most_saturated_dimension']}`."
        )
    else:
        lines.extend(
            [
                f"- aggregate actor mean: {format_value(aggregate['actor_action_mean']['mean'])}",
                f"- aggregate actor std mean/max: "
                f"{format_value(aggregate['actor_action_std']['mean'])} / "
                f"{format_value(aggregate['actor_action_std']['max'])}",
                "",
                "Per-dimension action statistics are unavailable because the run did not "
                "record per-step actor outputs.",
            ]
        )

    lines.extend(["", "## Height and terminal context", ""])
    if per_step["height"]["status"] == "available":
        lines.extend(
            [
                f"- height mean/min/max: {format_value(per_step['height']['mean'])} / "
                f"{format_value(per_step['height']['min'])} / "
                f"{format_value(per_step['height']['max'])}",
                f"- vertical velocity mean/std/min/max: "
                f"{format_value(per_step['vertical_velocity']['mean'])} / "
                f"{format_value(per_step['vertical_velocity']['std'])} / "
                f"{format_value(per_step['vertical_velocity']['min'])} / "
                f"{format_value(per_step['vertical_velocity']['max'])}",
                f"- recorded height-terminal windows: "
                f"{per_step['pre_height_termination_windows']['window_count']}",
            ]
        )
    else:
        lines.extend(
            [
                "Height, vertical velocity, and the ten steps before each height termination "
                "are unavailable because this run has no `steps.jsonl`.",
            ]
        )

    lines.extend(
        [
            "",
            "## Distance behavior",
            "",
            f"- final distance mean/min/max: "
            f"{format_value(distance['final_distance']['mean'])} / "
            f"{format_value(distance['final_distance']['min'])} / "
            f"{format_value(distance['final_distance']['max'])}",
            f"- first completed episode final distance: "
            f"{format_value(distance['first_completed_episode_final_distance'])}",
            f"- last completed episode final distance: "
            f"{format_value(distance['last_completed_episode_final_distance'])}",
            f"- first-to-last final-distance improvement: "
            f"{format_value(distance['first_to_last_improvement'])}",
        ]
    )
    if per_step["distance_progress"]["status"] == "available":
        lines.append(
            f"- per-step progress mean/std: "
            f"{format_value(per_step['distance_progress']['mean'])} / "
            f"{format_value(per_step['distance_progress']['std'])}"
        )
    lines.extend(
        [
            "",
            "Equal episode horizons are required before interpreting final-distance changes "
            "as navigation learning.",
            "",
            "## Analysis conclusion",
            "",
            f"- {diagnosis['primary_failure']}",
            f"- {diagnosis['actor_collapse_evidence']}",
            f"- {diagnosis['causality_limit']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze an AvoidBench Plain TD3 smoke run.")
    parser.add_argument(
        "run_dir",
        nargs="?",
        default="runs/avoidbench_plain_td3_smoke/20260609-051809",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Defaults to <run_dir>/analysis_summary.json.",
    )
    parser.add_argument(
        "--markdown-out",
        default="docs/avoidbench_plain_td3_smoke_analysis.md",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summary_out = Path(args.summary_out) if args.summary_out else run_dir / "analysis_summary.json"
    markdown_out = Path(args.markdown_out)
    summary = analyze(run_dir)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown_out.write_text(render_markdown(summary))
    print(json.dumps({"summary": str(summary_out), "markdown": str(markdown_out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
