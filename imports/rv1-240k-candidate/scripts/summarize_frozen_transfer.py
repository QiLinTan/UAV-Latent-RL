"""Summarize the frozen start-curriculum zero-shot transfer matrix.

Reads completed production evaluation artifacts; does not run or train a model.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from statistics import mean, median


ROOT = Path("analysis_outputs/startcurr_zero_shot_20261009")
TRAINING_MODE = "upper_dslpid_recurrent_multitree_terminal_v2_noroute_startcurr_v1_h8_k1"
SEEDS = (42, 43, 44, 45)


def read_json(path: Path):
    return json.loads(path.read_text())


def first_jsonl(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError(f"Expected one evaluation in {path}; got {len(rows)}")
    return rows[0]


def evaluated_run(model: str, stage: int, seed: int):
    directory = ROOT / f"{model}_stage{stage}_seed{seed}"
    protocol = read_json(directory / "protocol.json")
    if (protocol["model"], protocol["forced_stage"], protocol["training_seed"]) != (
        model, stage, seed
    ):
        raise ValueError(f"Protocol mismatch: {directory}")
    evaluation = first_jsonl(directory / "evaluation_metrics.jsonl")
    coverage = read_json(directory / "coverage" / "terminal_coverage_latest.json")
    collisions = read_json(directory / "collision" / "collision_windows_latest.json")
    episodes = evaluation["episodes"]
    coverage_episodes = coverage["episodes"]
    if len(episodes) != 20 or len(coverage_episodes) != 20:
        raise ValueError(f"Expected 20 episodes: {directory}")
    if any(a["episode"] != b["episode"] or a["done_reason"] != b["done_reason"]
           for a, b in zip(episodes, coverage_episodes)):
        raise ValueError(f"Episode/coverage mismatch: {directory}")
    outcomes = Counter(e["done_reason"] for e in episodes)
    entered = sum(e["entered_guidance_region"] for e in coverage_episodes)
    failed = [e for e in coverage_episodes if e["done_reason"] != "success"]
    classification = Counter(c["classification"]["code"] for c in collisions["collisions"])
    if sum(classification.values()) != outcomes["collision"]:
        raise ValueError(f"Collision diagnostic mismatch: {directory}")
    return {
        "model": model,
        "stage": stage,
        "training_seed": seed,
        "checkpoint_actor_sha256": protocol["actor_sha256"],
        "route_blocking_tree_count": protocol["expected_route_blocking_tree_count"],
        "episodes": len(episodes),
        "outcomes": dict(outcomes),
        "guidance_entered": entered,
        "fail_before_guidance": sum(not e["entered_guidance_region"] for e in failed),
        "fail_after_guidance": sum(e["entered_guidance_region"] for e in failed),
        "failed_minimum_distance_median": median(e["minimum_goal_distance_m"] for e in failed)
        if failed else None,
        "collision_classification": dict(classification),
        "mean_return": evaluation["summary"]["eval/return"],
        "mean_final_goal_distance_m": evaluation["summary"]["eval/final_goal_distance"],
        "tracking_position_rmse_m": evaluation["summary"]["eval/tracking_position_rmse"]
        if model == "full" else None,
        "tracking_velocity_rmse_mps": evaluation["summary"]["eval/tracking_velocity_rmse"]
        if model == "full" else None,
        "max_roll_pitch_rad": evaluation["summary"]["eval/max_roll_pitch"],
        "episode_outcomes": {str(e["episode"]): e["done_reason"] for e in episodes},
    }


def aggregate(runs):
    outcomes = Counter()
    classes = Counter()
    for row in runs:
        outcomes.update(row["outcomes"])
        classes.update(row["collision_classification"])
    return {
        "episodes": sum(row["episodes"] for row in runs),
        "outcomes": dict(outcomes),
        "guidance_entered": sum(row["guidance_entered"] for row in runs),
        "fail_before_guidance": sum(row["fail_before_guidance"] for row in runs),
        "fail_after_guidance": sum(row["fail_after_guidance"] for row in runs),
        "collision_classification": dict(classes),
        "mean_return_across_runs": mean(row["mean_return"] for row in runs),
        "mean_final_goal_distance_m": mean(row["mean_final_goal_distance_m"] for row in runs),
        "training_seed_success_counts": {str(row["training_seed"]): row["outcomes"].get("success", 0)
                                         for row in runs},
        "environment_seed_success_counts": {
            str(ep): sum(row["episode_outcomes"][str(ep)] == "success" for row in runs)
            for ep in range(20)
        },
    }


def regression(seed: int, chosen_step: int):
    path = Path("runs") / f"{TRAINING_MODE}_seed{seed}" / "evaluation_metrics.jsonl"
    periodic = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # The final duplicate step is an evaluation after restoring model_best.
        periodic.setdefault(row["step"], row)
    selected = periodic[chosen_step]
    last_step = max(periodic)
    last = periodic[last_step]
    later = [row for step, row in sorted(periodic.items()) if step > chosen_step]
    measure = lambda row, metric: row["summary"][metric]
    return {
        "training_seed": seed,
        "selected_checkpoint_step": chosen_step,
        "selected_start_level": selected["curriculum"]["start_curriculum_level"],
        "selected_raw_10_episode_success": measure(selected, "eval/success_rate"),
        "selected_raw_10_episode_collision": measure(selected, "eval/collision_rate"),
        "last_raw_step": last_step,
        "last_start_level": last["curriculum"]["start_curriculum_level"],
        "last_raw_10_episode_success": measure(last, "eval/success_rate"),
        "last_raw_10_episode_collision": measure(last, "eval/collision_rate"),
        "last_raw_10_episode_timeout": measure(last, "eval/done_reason_timeout_rate"),
        "worst_post_selection_success": min(
            (measure(row, "eval/success_rate") for row in later), default=None
        ),
        "post_selection_evaluations_below_50pct": sum(
            measure(row, "eval/success_rate") < 0.5 for row in later
        ),
        "post_selection_evaluations": len(later),
        "same_level_declines": [
            {"step": row["step"], "success": measure(row, "eval/success_rate")}
            for row in later
            if row["curriculum"]["start_curriculum_level"] ==
            selected["curriculum"]["start_curriculum_level"]
            and measure(row, "eval/success_rate") <
            measure(selected, "eval/success_rate") - 0.2
        ],
    }


def main():
    frozen = read_json(Path("checkpoints/startcurr_v1_frozen_20261009/manifest.json"))
    chosen = {int(seed): entry["selected_step"] for seed, entry in frozen["seeds"].items()}
    runs = [evaluated_run("k1", stage, seed)
            for stage in (1, 2, 3) for seed in SEEDS]
    runs += [evaluated_run("full", 0, seed) for seed in SEEDS]
    summary = {
        "protocol": {
            "training_seeds": SEEDS,
            "environment_seeds": list(range(20)),
            "zero_shot": True,
            "new_training_updates": 0,
            "original_start_distance_m": 7.0,
            "success_radius_3d_m": 0.2,
            "guidance_radius_3d_m": 1.2,
            "simple_obstacle_mix_probability": 0.0,
            "map_route_reachable": "20/20 at each stage in separate reset audit",
        },
        "runs": runs,
        "aggregates": {
            f"k1_stage{stage}": aggregate([r for r in runs if r["model"] == "k1"
                                            and r["stage"] == stage])
            for stage in (1, 2, 3)
        } | {"full_stage0": aggregate([r for r in runs if r["model"] == "full"])},
        "raw_training_regression": [regression(seed, chosen[seed]) for seed in SEEDS],
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for label, row in summary["aggregates"].items():
        print(label, row["outcomes"], "guidance", row["guidance_entered"],
              "pre_guidance_fail", row["fail_before_guidance"])


if __name__ == "__main__":
    main()
